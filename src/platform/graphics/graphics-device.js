import { version } from '../../core/core.js';
import { Debug } from '../../core/debug.js';
import { EventHandler } from '../../core/event-handler.js';
import { platform } from '../../core/platform.js';
import { now } from '../../core/time.js';
import { Vec2 } from '../../core/math/vec2.js';
import { Tracing } from '../../core/tracing.js';
import { Color } from '../../core/math/color.js';
import { TRACEID_TEXTURES } from '../../core/constants.js';
import {
    CULLFACE_BACK,
    CLEARFLAG_COLOR, CLEARFLAG_DEPTH,
    PRIMITIVE_POINTS, PRIMITIVE_TRIFAN, SEMANTIC_POSITION, TYPE_FLOAT32, PIXELFORMAT_111110F, PIXELFORMAT_RGBA16F, PIXELFORMAT_RGBA32F,
    DISPLAYFORMAT_LDR,
    semanticToLocation
} from './constants.js';
import { BlendState } from './blend-state.js';
import { DepthState } from './depth-state.js';
import { ScopeSpace } from './scope-space.js';
import { VertexBuffer } from './vertex-buffer.js';
import { VertexFormat } from './vertex-format.js';
import { StencilParameters } from './stencil-parameters.js';
import { DebugGraphics } from './debug-graphics.js';

/**
 * @import { Compute } from './compute.js'
 * @import { DEVICETYPE_WEBGL2, DEVICETYPE_WEBGPU } from './constants.js'
 * @import { DynamicBuffers } from './dynamic-buffers.js'
 * @import { GpuProfiler } from './gpu-profiler.js'
 * @import { IndexBuffer } from './index-buffer.js'
 * @import { RenderTarget } from './render-target.js'
 * @import { Shader } from './shader.js'
 * @import { Texture } from './texture.js'
 * @import { StorageBuffer } from './storage-buffer.js';
 */

const _tempSet = new Set();

/**
 * The graphics device manages the underlying graphics context. It is responsible for submitting
 * render state changes and graphics primitives to the hardware. A graphics device is tied to a
 * specific canvas HTML element. It is valid to have more than one canvas element per page and
 * create a new graphics device against each.
 *
 * @category Graphics
 */
class GraphicsDevice extends EventHandler {
    /**
     * Fired when the canvas is resized. The handler is passed the new width and height as number
     * parameters.
     *
     * @event
     * @example
     * graphicsDevice.on('resizecanvas', (width, height) => {
     *     console.log(`The canvas was resized to ${width}x${height}`);
     * });
     */

    /**
     * The canvas DOM element that provides the underlying WebGL context used by the graphics device.
     *
     * @type {HTMLCanvasElement}
     * @readonly
     */
    canvas;

    /**
     * The render target representing the main back-buffer.
     *
     * @type {RenderTarget|null}
     * @ignore
     */
    backBuffer = null;

    /**
     * The dimensions of the back buffer.
     *
     * @ignore
     */
    backBufferSize = new Vec2();

    /**
     * The pixel format of the back buffer. Typically PIXELFORMAT_RGBA8, PIXELFORMAT_BGRA8 or
     * PIXELFORMAT_RGB8.
     *
     * @ignore
     */
    backBufferFormat;

    /**
     * True if the back buffer should use anti-aliasing.
     *
     * @type {boolean}
     */
    backBufferAntialias = false;

    /**
     * True if the deviceType is WebGPU
     *
     * @type {boolean}
     * @readonly
     */
    isWebGPU = false;

    /**
     * True if the deviceType is WebGL2
     *
     * @type {boolean}
     * @readonly
     */
    isWebGL2 = false;

    /**
     * True if the back-buffer is using HDR format, which means that the browser will display the
     * rendered images in high dynamic range mode. This is true if the options.displayFormat is set
     * to {@link DISPLAYFORMAT_HDR} when creating the graphics device using
     * {@link createGraphicsDevice}, and HDR is supported by the device.
     */
    isHdr = false;

    /**
     * The scope namespace for shader attributes and variables.
     *
     * @type {ScopeSpace}
     * @readonly
     */
    scope;

    /**
     * The maximum number of indirect draw calls that can be used within a single frame. Used on
     * WebGPU only. This needs to be adjusted based on the maximum number of draw calls that can
     * be used within a single frame. Defaults to 1024.
     *
     * @type {number}
     */
    maxIndirectDrawCount = 1024;

    /**
     * The maximum supported texture anisotropy setting.
     *
     * @type {number}
     * @readonly
     */
    maxAnisotropy;

    /**
     * The maximum supported dimension of a cube map.
     *
     * @type {number}
     * @readonly
     */
    maxCubeMapSize;

    /**
     * The maximum supported dimension of a texture.
     *
     * @type {number}
     * @readonly
     */
    maxTextureSize;

    /**
     * The maximum supported dimension of a 3D texture (any axis).
     *
     * @type {number}
     * @readonly
     */
    maxVolumeSize;

    /**
     * The maximum supported number of color buffers attached to a render target.
     *
     * @type {number}
     * @readonly
     */
    maxColorAttachments = 1;

    /**
     * The highest shader precision supported by this graphics device. Can be 'hiphp', 'mediump' or
     * 'lowp'.
     *
     * @type {string}
     * @readonly
     */
    precision;

    /**
     * The number of hardware anti-aliasing samples used by the frame buffer.
     *
     * @readonly
     * @type {number}
     */
    samples;

    /**
     * The maximum supported number of hardware anti-aliasing samples.
     *
     * @readonly
     * @type {number}
     */
    maxSamples = 1;

    /**
     * True if the main framebuffer contains stencil attachment.
     *
     * @ignore
     * @type {boolean}
     */
    supportsStencil;

    /**
     * True if the device supports compute shaders.
     *
     * @readonly
     * @type {boolean}
     */
    supportsCompute = false;

    /**
     * True if the device can read from StorageTexture in the compute shader. By default, the
     * storage texture can be only used with the write operation.
     * When a shader uses this feature, it's recommended to use a `requires` directive to signal the
     * potential for non-portability at the top of the WGSL shader code:
     * ```javascript
     * requires readonly_and_readwrite_storage_textures;
     * ```
     *
     * @readonly
     * @type {boolean}
     */
    supportsStorageTextureRead = false;

    /**
     * Currently active render target.
     *
     * @type {RenderTarget|null}
     * @ignore
     */
    renderTarget = null;

    /**
     * Array of objects that need to be re-initialized after a context restore event
     *
     * @type {Shader[]}
     * @ignore
     */
    shaders = [];

    /**
     * An array of currently created textures.
     *
     * @type {Texture[]}
     * @ignore
     */
    textures = [];

    /**
     * A set of currently created render targets.
     *
     * @type {Set<RenderTarget>}
     * @ignore
     */
    targets = new Set();

    /**
     * A version number that is incremented every frame. This is used to detect if some object were
     * invalidated.
     *
     * @type {number}
     * @ignore
     */
    renderVersion = 0;

    /**
     * Index of the currently active render pass.
     *
     * @type {number}
     * @ignore
     */
    renderPassIndex;

    /** @type {boolean} */
    insideRenderPass = false;

    /**
     * True if the device supports uniform buffers.
     *
     * @type {boolean}
     * @ignore
     */
    supportsUniformBuffers = false;

    /**
     * True if the device supports clip distances (WebGPU only). Clip distances allow you to restrict
     * primitives' clip volume with user-defined half-spaces in the output of vertex stage.
     *
     * @type {boolean}
     */
    supportsClipDistances = false;

    /**
     * True if 32-bit floating-point textures can be used as a frame buffer.
     *
     * @type {boolean}
     * @readonly
     */
    textureFloatRenderable;

    /**
     * True if 16-bit floating-point textures can be used as a frame buffer.
     *
     * @type {boolean}
     * @readonly
     */
    textureHalfFloatRenderable;

    /**
     * True if small-float textures with format {@link PIXELFORMAT_111110F} can be used as a frame
     * buffer. This is always true on WebGL2, but optional on WebGPU device.
     *
     * @type {boolean}
     * @readonly
     */
    textureRG11B10Renderable = false;

    /**
     * True if filtering can be applied when sampling float textures.
     *
     * @type {boolean}
     * @readonly
     */
    textureFloatFilterable = false;

    /**
     * A vertex buffer representing a quad.
     *
     * @type {VertexBuffer}
     * @ignore
     */
    quadVertexBuffer;

    /**
     * An object representing current blend state
     *
     * @ignore
     */
    blendState = new BlendState();

    /**
     * The current depth state.
     *
     * @ignore
     */
    depthState = new DepthState();

    /**
     * True if stencil is enabled and stencilFront and stencilBack are used
     *
     * @ignore
     */
    stencilEnabled = false;

    /**
     * The current front stencil parameters.
     *
     * @ignore
     */
    stencilFront = new StencilParameters();

    /**
     * The current back stencil parameters.
     *
     * @ignore
     */
    stencilBack = new StencilParameters();

    /**
     * The dynamic buffer manager.
     *
     * @type {DynamicBuffers}
     * @ignore
     */
    dynamicBuffers;

    /**
     * The GPU profiler.
     *
     * @type {GpuProfiler}
     */
    gpuProfiler;

    defaultClearOptions = {
        color: [0, 0, 0, 1],
        depth: 1,
        stencil: 0,
        flags: CLEARFLAG_COLOR | CLEARFLAG_DEPTH
    };

    /**
     * The current client rect.
     *
     * @type {{ width: number, height: number }}
     * @ignore
     */
    clientRect = {
        width: 0,
        height: 0
    };

    /**
     * A very heavy handed way to force all shaders to be rebuilt. Avoid using as much as possible.
     *
     * @type {boolean}
     * @ignore
     */
    _shadersDirty = false;

    /**
     * A list of shader defines based on the capabilities of the device.
     *
     * @type {Map<string, string>}
     * @ignore
     */
    capsDefines = new Map();

    /**
     * A set of maps to clear at the end of the frame.
     *
     * @type {Set<Map>}
     * @ignore
     */
    mapsToClear = new Set();

    static EVENT_RESIZE = 'resizecanvas';

    constructor(canvas, options) {
        super();

        this.canvas = canvas;
        if ('setAttribute' in canvas) {
            canvas.setAttribute('data-engine', `PlayCanvas ${version}`);
        }

        // copy options and handle defaults
        this.initOptions = { ...options };
        this.initOptions.alpha ??= true;
        this.initOptions.depth ??= true;
        this.initOptions.stencil ??= true;
        this.initOptions.antialias ??= true;
        this.initOptions.powerPreference ??= 'high-performance';
        this.initOptions.displayFormat ??= DISPLAYFORMAT_LDR;

        // Some devices window.devicePixelRatio can be less than one
        // eg Oculus Quest 1 which returns a window.devicePixelRatio of 0.8
        this._maxPixelRatio = platform.browser ? Math.min(1, window.devicePixelRatio) : 1;

        this.buffers = [];

        this._vram = {
            // #if _PROFILER
            texShadow: 0,
            texAsset: 0,
            texLightmap: 0,
            // #endif
            tex: 0,
            vb: 0,
            ib: 0,
            ub: 0,
            sb: 0
        };

        this._shaderStats = {
            vsCompiled: 0,
            fsCompiled: 0,
            linked: 0,
            materialShaders: 0,
            compileTime: 0
        };

        this.initializeContextCaches();

        // Profiler stats
        this._drawCallsPerFrame = 0;
        this._shaderSwitchesPerFrame = 0;

        this._primsPerFrame = [];
        for (let i = PRIMITIVE_POINTS; i <= PRIMITIVE_TRIFAN; i++) {
            this._primsPerFrame[i] = 0;
        }
        this._renderTargetCreationTime = 0;

        // Create the ScopeNamespace for shader attributes and variables
        this.scope = new ScopeSpace('Device');

        this.textureBias = this.scope.resolve('textureBias');
        this.textureBias.setValue(0.0);
    }

    /**
     * Function that executes after the device has been created.
     */
    postInit() {

        // create quad vertex buffer
        const vertexFormat = new VertexFormat(this, [
            { semantic: SEMANTIC_POSITION, components: 2, type: TYPE_FLOAT32 }
        ]);
        const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        this.quadVertexBuffer = new VertexBuffer(this, vertexFormat, 4, {
            data: positions
        });
    }

    /**
     * Initialize the map of device capabilities, which are supplied to shaders as defines.
     *
     * @ignore
     */
    initCapsDefines() {
        const { capsDefines } = this;
        capsDefines.clear();
        if (this.textureFloatFilterable) capsDefines.set('CAPS_TEXTURE_FLOAT_FILTERABLE', '');
        if (this.textureFloatRenderable) capsDefines.set('CAPS_TEXTURE_FLOAT_RENDERABLE', '');
    }

    /**
     * Destroy the graphics device.
     */
    destroy() {
        // fire the destroy event.
        // textures and other device resources may destroy themselves in response.
        this.fire('destroy');

        this.quadVertexBuffer?.destroy();
        this.quadVertexBuffer = null;

        this.dynamicBuffers?.destroy();
        this.dynamicBuffers = null;

        this.gpuProfiler?.destroy();
        this.gpuProfiler = null;
    }

    onDestroyShader(shader) {
        this.fire('destroy:shader', shader);

        const idx = this.shaders.indexOf(shader);
        if (idx !== -1) {
            this.shaders.splice(idx, 1);
        }
    }

    // executes after the extended classes have executed their destroy function
    postDestroy() {
        this.scope = null;
        this.canvas = null;
    }

    /**
     * Called when the device context was lost. It releases all context related resources.
     *
     * @ignore
     */
    loseContext() {

        this.contextLost = true;

        // force the back-buffer to be recreated on restore
        this.backBufferSize.set(-1, -1);

        // release textures
        for (const texture of this.textures) {
            texture.loseContext();
        }

        // release vertex and index buffers
        for (const buffer of this.buffers) {
            buffer.loseContext();
        }

        // Reset all render targets so they'll be recreated as required.
        // TODO: a solution for the case where a render target contains something
        // that was previously generated that needs to be re-rendered.
        for (const target of this.targets) {
            target.loseContext();
        }

        this.gpuProfiler?.loseContext();
    }

    /**
     * Called when the device context is restored. It reinitializes all context related resources.
     *
     * @ignore
     */
    restoreContext() {

        this.contextLost = false;

        this.initializeRenderState();
        this.initializeContextCaches();

        // Recreate buffer objects and reupload buffer data to the GPU
        for (const buffer of this.buffers) {
            buffer.unlock();
        }

        this.gpuProfiler?.restoreContext?.();
    }

    // don't stringify GraphicsDevice to JSON by JSON.stringify
    toJSON(key) {
        return undefined;
    }

    initializeContextCaches() {
        this.vertexBuffers = [];
        this.shader = null;
        this.shaderValid = undefined;
        this.shaderAsyncCompile = false;
        this.renderTarget = null;
    }

    initializeRenderState() {

        this.blendState = new BlendState();
        this.depthState = new DepthState();
        this.cullMode = CULLFACE_BACK;

        // Cached viewport and scissor dimensions
        this.vx = this.vy = this.vw = this.vh = 0;
        this.sx = this.sy = this.sw = this.sh = 0;

        this.blendColor = new Color(0, 0, 0, 0);
    }

    /**
     * Sets the specified stencil state. If both stencilFront and stencilBack are null, stencil
     * operation is disabled.
     *
     * @param {StencilParameters} [stencilFront] - The front stencil parameters. Defaults to
     * {@link StencilParameters.DEFAULT} if not specified.
     * @param {StencilParameters} [stencilBack] - The back stencil parameters. Defaults to
     * {@link StencilParameters.DEFAULT} if not specified.
     */
    setStencilState(stencilFront, stencilBack) {
        Debug.assert(false);
    }

    /**
     * Sets the specified blend state.
     *
     * @param {BlendState} blendState - New blend state.
     */
    setBlendState(blendState) {
        Debug.assert(false);
    }

    /**
     * Sets the constant blend color and alpha values used with {@link BLENDMODE_CONSTANT} and
     * {@link BLENDMODE_ONE_MINUS_CONSTANT} factors specified in {@link BlendState}. Defaults to
     * [0, 0, 0, 0].
     *
     * @param {number} r - The value for red.
     * @param {number} g - The value for green.
     * @param {number} b - The value for blue.
     * @param {number} a - The value for alpha.
     */
    setBlendColor(r, g, b, a) {
        Debug.assert(false);
    }

    /**
     * Sets the specified depth state.
     *
     * @param {DepthState} depthState - New depth state.
     */
    setDepthState(depthState) {
        Debug.assert(false);
    }

    /**
     * Controls how triangles are culled based on their face direction. The default cull mode is
     * {@link CULLFACE_BACK}.
     *
     * @param {number} cullMode - The cull mode to set. Can be:
     *
     * - {@link CULLFACE_NONE}
     * - {@link CULLFACE_BACK}
     * - {@link CULLFACE_FRONT}
     */
    setCullMode(cullMode) {
        Debug.assert(false);
    }

    /**
     * Sets the specified render target on the device. If null is passed as a parameter, the back
     * buffer becomes the current target for all rendering operations.
     *
     * @param {RenderTarget|null} renderTarget - The render target to activate.
     * @example
     * // Set a render target to receive all rendering output
     * device.setRenderTarget(renderTarget);
     *
     * // Set the back buffer to receive all rendering output
     * device.setRenderTarget(null);
     */
    setRenderTarget(renderTarget) {
        this.renderTarget = renderTarget;
    }

    /**
     * Sets the current vertex buffer on the graphics device. For subsequent draw calls, the
     * specified vertex buffer(s) will be used to provide vertex data for any primitives.
     *
     * @param {VertexBuffer} vertexBuffer - The vertex buffer to assign to the device.
     * @ignore
     */
    setVertexBuffer(vertexBuffer) {

        if (vertexBuffer) {
            this.vertexBuffers.push(vertexBuffer);
        }
    }

    /**
     * Clears the vertex buffer set on the graphics device. This is called automatically by the
     * renderer.
     * @ignore
     */
    clearVertexBuffer() {
        this.vertexBuffers.length = 0;
    }

    /**
     * Retrieves the available slot in the {@link indirectDrawBuffer} used for indirect rendering,
     * which can be utilized by a {@link Compute} shader to generate indirect draw parameters and by
     * {@link MeshInstance#setIndirect} to configure indirect draw calls.
     *
     * @returns {number} - The slot used for indirect rendering.
     */
    getIndirectDrawSlot() {
        return 0;
    }

    /**
     * Returns the buffer used to store arguments for indirect draw calls. The size of the buffer is
     * controlled by the {@link maxIndirectDrawCount} property. This buffer can be passed to a
     * {@link Compute} shader along with a slot obtained by calling {@link getIndirectDrawSlot}, in
     * order to prepare indirect draw parameters. Also see {@link MeshInstance#setIndirect}.
     *
     * Only available on WebGPU, returns null on other platforms.
     *
     * @type {StorageBuffer|null}
     */
    get indirectDrawBuffer() {
        return null;
    }

    /**
     * Queries the currently set render target on the device.
     *
     * @returns {RenderTarget} The current render target.
     * @example
     * // Get the current render target
     * const renderTarget = device.getRenderTarget();
     */
    getRenderTarget() {
        return this.renderTarget;
    }

    /**
     * Initialize render target before it can be used.
     *
     * @param {RenderTarget} target - The render target to be initialized.
     * @ignore
     */
    initRenderTarget(target) {

        if (target.initialized) return;

        // #if _PROFILER
        const startTime = now();
        this.fire('fbo:create', {
            timestamp: startTime,
            target: this
        });
        // #endif

        target.init();
        this.targets.add(target);

        // #if _PROFILER
        this._renderTargetCreationTime += now() - startTime;
        // #endif
    }

    /**
     * Submits a graphical primitive to the hardware for immediate rendering.
     *
     * @param {object} primitive - Primitive object describing how to submit current vertex/index
     * buffers.
     * @param {number} primitive.type - The type of primitive to render. Can be:
     *
     * - {@link PRIMITIVE_POINTS}
     * - {@link PRIMITIVE_LINES}
     * - {@link PRIMITIVE_LINELOOP}
     * - {@link PRIMITIVE_LINESTRIP}
     * - {@link PRIMITIVE_TRIANGLES}
     * - {@link PRIMITIVE_TRISTRIP}
     * - {@link PRIMITIVE_TRIFAN}
     *
     * @param {number} primitive.base - The offset of the first index or vertex to dispatch in the
     * draw call.
     * @param {number} primitive.count - The number of indices or vertices to dispatch in the draw
     * call.
     * @param {boolean} [primitive.indexed] - True to interpret the primitive as indexed, thereby
     * using the currently set index buffer and false otherwise.
     * @param {IndexBuffer} [indexBuffer] - The index buffer to use for the draw call.
     * @param {number} [numInstances] - The number of instances to render when using instancing.
     * Defaults to 1.
     * @param {number} [indirectSlot] - The slot of the indirect buffer to use for the draw call.
     * @param {boolean} [first] - True if this is the first draw call in a sequence of draw calls.
     * When set to true, vertex and index buffers related state is set up. Defaults to true.
     * @param {boolean} [last] - True if this is the last draw call in a sequence of draw calls.
     * When set to true, vertex and index buffers related state is cleared. Defaults to true.
     * @example
     * // Render a single, unindexed triangle
     * device.draw({
     *     type: pc.PRIMITIVE_TRIANGLES,
     *     base: 0,
     *     count: 3,
     *     indexed: false
     * });
     *
     * @ignore
     */
    draw(primitive, indexBuffer, numInstances, indirectSlot, first = true, last = true) {
        Debug.assert(false);
    }

    /**
     * Reports whether a texture source is a canvas, image, video or ImageBitmap.
     *
     * @param {*} texture - Texture source data.
     * @returns {boolean} True if the texture is a canvas, image, video or ImageBitmap and false
     * otherwise.
     * @ignore
     */
    _isBrowserInterface(texture) {
        return this._isImageBrowserInterface(texture) ||
                this._isImageCanvasInterface(texture) ||
                this._isImageVideoInterface(texture);
    }

    _isImageBrowserInterface(texture) {
        return (typeof ImageBitmap !== 'undefined' && texture instanceof ImageBitmap) ||
               (typeof HTMLImageElement !== 'undefined' && texture instanceof HTMLImageElement);
    }

    _isImageCanvasInterface(texture) {
        return (typeof HTMLCanvasElement !== 'undefined' && texture instanceof HTMLCanvasElement);
    }

    _isImageVideoInterface(texture) {
        return (typeof HTMLVideoElement !== 'undefined' && texture instanceof HTMLVideoElement);
    }

    /**
     * Sets the width and height of the canvas, then fires the `resizecanvas` event. Note that the
     * specified width and height values will be multiplied by the value of
     * {@link GraphicsDevice#maxPixelRatio} to give the final resultant width and height for the
     * canvas.
     *
     * @param {number} width - The new width of the canvas.
     * @param {number} height - The new height of the canvas.
     * @ignore
     */
    resizeCanvas(width, height) {
        const pixelRatio = Math.min(this._maxPixelRatio, platform.browser ? window.devicePixelRatio : 1);
        const w = Math.floor(width * pixelRatio);
        const h = Math.floor(height * pixelRatio);
        if (w !== this.canvas.width || h !== this.canvas.height) {
            this.setResolution(w, h);
        }
    }

    /**
     * Sets the width and height of the canvas, then fires the `resizecanvas` event. Note that the
     * value of {@link GraphicsDevice#maxPixelRatio} is ignored.
     *
     * @param {number} width - The new width of the canvas.
     * @param {number} height - The new height of the canvas.
     * @ignore
     */
    setResolution(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.fire(GraphicsDevice.EVENT_RESIZE, width, height);
    }

    update() {
        this.updateClientRect();
    }

    updateClientRect() {
        if (platform.worker) {
            // Web Workers don't do page layout, so getBoundingClientRect is not available
            this.clientRect.width = this.canvas.width;
            this.clientRect.height = this.canvas.height;
        } else {
            const rect = this.canvas.getBoundingClientRect();
            this.clientRect.width = rect.width;
            this.clientRect.height = rect.height;
        }
    }

    /**
     * Width of the back buffer in pixels.
     *
     * @type {number}
     */
    get width() {
        return this.canvas.width;
    }

    /**
     * Height of the back buffer in pixels.
     *
     * @type {number}
     */
    get height() {
        return this.canvas.height;
    }

    /**
     * Sets whether the device is currently in fullscreen mode.
     *
     * @type {boolean}
     */
    set fullscreen(fullscreen) {
        Debug.error('GraphicsDevice.fullscreen is not implemented on current device.');
    }

    /**
     * Gets whether the device is currently in fullscreen mode.
     *
     * @type {boolean}
     */
    get fullscreen() {
        Debug.error('GraphicsDevice.fullscreen is not implemented on current device.');
        return false;
    }

    /**
     * Sets the maximum pixel ratio.
     *
     * @type {number}
     */
    set maxPixelRatio(ratio) {
        this._maxPixelRatio = ratio;
    }

    /**
     * Gets the maximum pixel ratio.
     *
     * @type {number}
     */
    get maxPixelRatio() {
        return this._maxPixelRatio;
    }

    /**
     * Gets the type of the device. Can be:
     *
     * - {@link DEVICETYPE_WEBGL2}
     * - {@link DEVICETYPE_WEBGPU}
     *
     * @type {DEVICETYPE_WEBGL2|DEVICETYPE_WEBGPU}
     */
    get deviceType() {
        return this._deviceType;
    }

    startRenderPass(renderPass) {
    }

    endRenderPass(renderPass) {
    }

    startComputePass(name) {
    }

    endComputePass() {
    }

    /**
     * Function which executes at the start of the frame. This should not be called manually, as
     * it is handled by the AppBase instance.
     *
     * @ignore
     */
    frameStart() {
        this.renderPassIndex = 0;
        this.renderVersion++;

        Debug.call(() => {

            // log out all loaded textures, sorted by gpu memory size
            if (Tracing.get(TRACEID_TEXTURES)) {
                const textures = this.textures.slice();
                textures.sort((a, b) => b.gpuSize - a.gpuSize);
                Debug.log(`Textures: ${textures.length}`);
                let textureTotal = 0;
                textures.forEach((texture, index) => {
                    const textureSize  = texture.gpuSize;
                    textureTotal += textureSize;
                    Debug.log(`${index}. ${texture.name} ${texture.width}x${texture.height} VRAM: ${(textureSize / 1024 / 1024).toFixed(2)} MB`);
                });
                Debug.log(`Total: ${(textureTotal / 1024 / 1024).toFixed(2)}MB`);
            }
        });
    }

    /**
     * Function which executes at the end of the frame. This should not be called manually, as it is
     * handled by the AppBase instance.
     *
     * @ignore
     */
    frameEnd() {
        // clear all maps scheduled for end of frame clearing
        this.mapsToClear.forEach(map => map.clear());
        this.mapsToClear.clear();
    }

    /**
     * Dispatch multiple compute shaders inside a single compute shader pass.
     *
     * @param {Array<Compute>} computes - An array of compute shaders to dispatch.
     * @param {string} [name] - The name of the dispatch, used for debugging and reporting only.
     */
    computeDispatch(computes, name = 'Unnamed') {
    }

    /**
     * Get a renderable HDR pixel format supported by the graphics device.
     *
     * Note:
     *
     * - When the `filterable` parameter is set to false, this function returns one of the supported
     * formats on the majority of devices apart from some very old iOS and Android devices (99%).
     * - When the `filterable` parameter is set to true, the function returns a format on a
     * considerably lower number of devices (70%).
     *
     * @param {number[]} [formats] - An array of pixel formats to check for support. Can contain:
     *
     * - {@link PIXELFORMAT_111110F}
     * - {@link PIXELFORMAT_RGBA16F}
     * - {@link PIXELFORMAT_RGBA32F}
     *
     * @param {boolean} [filterable] - If true, the format also needs to be filterable. Defaults to
     * true.
     * @param {number} [samples] - The number of samples to check for. Some formats are not
     * compatible with multi-sampling, for example {@link PIXELFORMAT_RGBA32F} on WebGPU platform.
     * Defaults to 1.
     * @returns {number|undefined} The first supported renderable HDR format or undefined if none is
     * supported.
     */
    getRenderableHdrFormat(formats = [PIXELFORMAT_111110F, PIXELFORMAT_RGBA16F, PIXELFORMAT_RGBA32F], filterable = true, samples = 1) {
        for (let i = 0; i < formats.length; i++) {
            const format = formats[i];
            switch (format) {

                case PIXELFORMAT_111110F: {
                    if (this.textureRG11B10Renderable) {
                        return format;
                    }
                    break;
                }

                case PIXELFORMAT_RGBA16F:
                    if (this.textureHalfFloatRenderable) {
                        return format;
                    }
                    break;

                case PIXELFORMAT_RGBA32F:

                    // on WebGPU platform, RGBA32F is not compatible with multi-sampling
                    if (this.isWebGPU && samples > 1) {
                        continue;
                    }

                    if (this.textureFloatRenderable && (!filterable || this.textureFloatFilterable)) {
                        return format;
                    }
                    break;
            }
        }
        return undefined;
    }

    /**
     * Validate that all attributes required by the shader are present in the currently assigned
     * vertex buffers.
     *
     * @param {Shader} shader - The shader to validate.
     * @param {VertexFormat} vb0Format - The format of the first vertex buffer.
     * @param {VertexFormat} vb1Format - The format of the second vertex buffer.
     * @protected
     */
    validateAttributes(shader, vb0Format, vb1Format) {

        Debug.call(() => {

            // add all attribute locations from vertex formats to the set
            _tempSet.clear();
            vb0Format?.elements.forEach(element => _tempSet.add(semanticToLocation[element.name]));
            vb1Format?.elements.forEach(element => _tempSet.add(semanticToLocation[element.name]));

            // every location shader needs must be in the vertex buffer
            for (const [location, name] of shader.attributes) {
                if (!_tempSet.has(location)) {
                    Debug.errorOnce(`Vertex attribute [${name}] at location ${location} required by the shader is not present in the currently assigned vertex buffers, while rendering [${DebugGraphics.toString()}]`, {
                        shader,
                        vb0Format,
                        vb1Format
                    });
                }
            }
        });
    }
}

export { GraphicsDevice };
