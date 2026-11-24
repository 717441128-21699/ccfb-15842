/**
 * WebGL2 图像增强流水线。
 *
 * 与 CPU 参考实现（core/enhancement.ts）算法对应，全部在 GPU 完成：
 *  S1 矩形化：鼻孔四边形/相似变换 -> 顶点着色器透视采样到 256x192 标准画布；
 *  S2 反光掩膜：片元着色器计算局部均值（用预计算的模糊 mip）+ 亮异常阈值；
 *  S3 扩散修复：反复乒乓渲染，掩膜像素取邻域均值，迭代收敛填补亮斑；
 *  S4 光照归一化：大尺度高斯背景相减 + 局部标准差标准化；
 *  S5 输出 8 位灰度到 Uint8Array 供 Worker 提特征。
 *
 * 不支持 WebGL2 时由调用方回退到 enhanceCpu。
 */

import { GrayImage, Point, NostrilPair, findHomography, invertMat3 } from '../core/geometry';
import { CANON_W, CANON_H } from '../core/synthesis';
import { similarityQuadFromNostrils } from '../core/enhancement';

const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
uniform vec4 u_quad; // 源图四边形（标准画布四角对应的源坐标，归一化 0..1）
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// 全屏片元：按 u_homography 把标准画布 uv 反映射到源 uv 采样。
const RECTIFY_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform mat3 u_invHom;
void main() {
  vec3 p = u_invHom * vec3(v_uv, 1.0);
  vec2 suv = p.xy / p.z;
  if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
    outColor = vec4(0.0);
    return;
  }
  outColor = texture(u_src, suv);
}`;

const SPEC_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform vec2 u_texel;
void main() {
  float c = texture(u_src, v_uv).r;
  // 15x15 局部均值（反光检测半径 7）。
  float sum = 0.0;
  for (int dy = -7; dy <= 7; dy++) {
    for (int dx = -7; dx <= 7; dx++) {
      sum += texture(u_src, v_uv + vec2(float(dx), float(dy)) * u_texel).r;
    }
  }
  float local = sum / 225.0;
  float spec = (c >= 0.855 && c - local >= 0.102) ? 1.0 : 0.0;
  outColor = vec4(c, spec, local, 1.0);
}`;

const INPAINT_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;   // r=灰度 g=反光掩膜(1=需修复)
uniform vec2 u_texel;
void main() {
  vec4 cur = texture(u_src, v_uv);
  if (cur.g < 0.5) {
    // 非反光像素：保留灰度，掩膜标记清零，供后续光照归一化只看 r 通道。
    outColor = vec4(cur.r, 0.0, 0.0, 1.0);
    return;
  }
  float sum = 0.0;
  sum += texture(u_src, v_uv + vec2( 1.0, 0.0) * u_texel).r;
  sum += texture(u_src, v_uv + vec2(-1.0, 0.0) * u_texel).r;
  sum += texture(u_src, v_uv + vec2( 0.0, 1.0) * u_texel).r;
  sum += texture(u_src, v_uv + vec2( 0.0,-1.0) * u_texel).r;
  outColor = vec4(sum * 0.25, 1.0, 0.0, 1.0);
}`;

const NORM_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_src;
uniform vec2 u_texel;
// 大尺度背景（半径 31 的盒模糊近似高斯 sigma~64 的低频）。
void main() {
  float bg = 0.0;
  const int R = 15;
  float norm = 0.0;
  for (int dy = -R; dy <= R; dy++) {
    for (int dx = -R; dx <= R; dx++) {
      bg += texture(u_src, v_uv + vec2(float(dx), float(dy)) * u_texel).r;
      norm += 1.0;
    }
  }
  bg /= norm;
  float c = texture(u_src, v_uv).r;
  // 高频 = 原图 - 大尺度背景，再做温和增益映射到 0..1。
  float hp = c - bg;
  float mapped = clamp(0.5 + hp * 2.2, 0.0, 1.0);
  outColor = vec4(vec3(mapped), 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error('shader compile: ' + gl.getShaderInfoLog(sh));
  }
  return sh;
}

function program(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('program link: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

export class GlEnhancer {
  private gl: WebGL2RenderingContext;
  private quadVao: WebGLVertexArrayObject;
  private progRectify: WebGLProgram;
  private progSpec: WebGLProgram;
  private progInpaint: WebGLProgram;
  private progNorm: WebGLProgram;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 不可用');
    this.gl = gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]), // 全屏三角
      gl.STATIC_DRAW,
    );
    this.quadVao = vao;
    this.progRectify = program(gl, VERT, RECTIFY_FRAG);
    this.progSpec = program(gl, VERT, SPEC_FRAG);
    this.progInpaint = program(gl, VERT, INPAINT_FRAG);
    this.progNorm = program(gl, VERT, NORM_FRAG);
  }

  static isSupported(): boolean {
    try {
      const c = document.createElement('canvas');
      return !!c.getContext('webgl2');
    } catch {
      return false;
    }
  }

  private makeTarget(w: number, h: number): { tex: WebGLTexture; fbo: WebGLFramebuffer } {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo };
  }

  private uploadSource(img: ImageData): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private bindFullscreen(prog: WebGLProgram): void {
    const gl = this.gl;
    gl.useProgram(prog);
    gl.bindVertexArray(this.quadVao);
    const loc = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }

  /**
   * 完整增强。nostrils 给定时走相似变换矩形化，否则用检测四边形。
   * 返回标准画布灰度（256x192）与反光占比。
   */
  enhance(
    img: ImageData,
    opts: { quad?: Point[]; nostrils?: NostrilPair; inpaintIterations?: number },
  ): { gray: GrayImage; specularRatio: number } {
    const gl = this.gl;
    const srcTex = this.uploadSource(img);
    const W = CANON_W;
    const H = CANON_H;

    // S1 矩形化到标准画布。
    const rect = this.makeTarget(W, H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, rect.fbo);
    gl.viewport(0, 0, W, H);
    this.bindFullscreen(this.progRectify);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(gl.getUniformLocation(this.progRectify, 'u_src'), 0);
    const invHom = this.quadToInverseHom(img.width, img.height, opts);
    gl.uniformMatrix3fv(gl.getUniformLocation(this.progRectify, 'u_invHom'), false, invHom);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // S2 反光掩膜。
    const spec = this.makeTarget(W, H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, spec.fbo);
    this.bindFullscreen(this.progSpec);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, rect.tex);
    gl.uniform1i(gl.getUniformLocation(this.progSpec, 'u_src'), 0);
    gl.uniform2f(gl.getUniformLocation(this.progSpec, 'u_texel'), 1 / W, 1 / H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // S3 扩散修复（乒乓）。
    let cur = spec;
    const iters = opts.inpaintIterations ?? 80;
    let ping = this.makeTarget(W, H);
    for (let i = 0; i < iters; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, ping.fbo);
      this.bindFullscreen(this.progInpaint);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, cur.tex);
      gl.uniform1i(gl.getUniformLocation(this.progInpaint, 'u_src'), 0);
      gl.uniform2f(gl.getUniformLocation(this.progInpaint, 'u_texel'), 1 / W, 1 / H);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      [cur, ping] = [ping, cur];
    }

    // S4 光照归一化。
    const out = this.makeTarget(W, H);
    gl.bindFramebuffer(gl.FRAMEBUFFER, out.fbo);
    this.bindFullscreen(this.progNorm);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cur.tex);
    gl.uniform1i(gl.getUniformLocation(this.progNorm, 'u_src'), 0);
    gl.uniform2f(gl.getUniformLocation(this.progNorm, 'u_texel'), 1 / W, 1 / H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // S5 读回。
    const pixels = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const gray = new GrayImage(W, H);
    // WebGL 原点在左下，翻转 y。
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const src = (H - 1 - y) * W * 4 + x * 4;
        gray.data[y * W + x] = pixels[src];
      }
    }
    // 反光占比：从 S2 目标读回 g 通道。
    gl.bindFramebuffer(gl.FRAMEBUFFER, spec.fbo);
    const specPx = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, specPx);
    let specCount = 0;
    for (let i = 1; i < specPx.length; i += 4) if (specPx[i] > 127) specCount++;
    const specularRatio = specCount / (W * H);

    // 清理 GPU 资源。
    gl.deleteTexture(srcTex);
    [rect, spec, ping, out].forEach((t) => {
      gl.deleteTexture(t.tex);
      gl.deleteFramebuffer(t.fbo);
    });
    return { gray, specularRatio };
  }

  /**
   * 计算「标准画布归一化 uv -> 源图归一化 uv」的 3x3 逆单应（列主序给 WebGL）。
   */
  private quadToInverseHom(
    srcW: number,
    srcH: number,
    opts: { quad?: Point[]; nostrils?: NostrilPair },
  ): Float32Array {
    let quad: Point[];
    if (opts.nostrils) {
      quad = similarityQuadFromNostrils(opts.nostrils);
    } else if (opts.quad) {
      quad = opts.quad;
    } else {
      // 整个源图。
      quad = [
        { x: 0, y: 0 },
        { x: srcW - 1, y: 0 },
        { x: srcW - 1, y: srcH - 1 },
        { x: 0, y: srcH - 1 },
      ];
    }
    // 源四边形（像素）-> 标准四角（像素）的单应，再转归一化并求逆。
    // 直接在归一化坐标求解：dst 为标准画布四角(0..1)，src 为四边形/图像尺寸。
    const srcNorm = quad.map((p) => ({ x: p.x / srcW, y: p.y / srcH }));
    const dstNorm = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ];
    const fwd = findHomography(srcNorm, dstNorm);
    const inv = invertMat3(fwd);
    // WebGL 列主序。
    return new Float32Array([
      inv[0], inv[3], inv[6],
      inv[1], inv[4], inv[7],
      inv[2], inv[5], inv[8],
    ]);
  }
}
