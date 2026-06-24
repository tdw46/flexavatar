import * as ort from "onnxruntime-web/wasm";

const ORT_WASM_PATHS = {
  mjs: "/ort/ort-wasm-simd-threaded.asyncify.mjs",
  wasm: "/ort/ort-wasm-simd-threaded.asyncify.wasm",
};

const ORT_UPSCALERS = {
  artcnn: {
    path: "/models/upscalers/artcnn/ArtCNN_C4F32.onnx",
    kind: "artcnn-rgb-luma",
  },
  artcnnDs: {
    path: "/models/upscalers/artcnn/ArtCNN_C4F32_DS.onnx",
    kind: "artcnn-rgb-luma",
  },
  animejanai: {
    path: "/models/upscalers/animejanai/AnimeJaNai_HD_V3.1Sharp1_Performance.onnx",
    kind: "rgb-f16",
  },
};

const REAL_CUGAN_MODEL_URL =
  "https://huggingface.co/shammisw/real-cugan-tensorflowjs/resolve/main/real-cugan-models/realcugan/2x-conservative-64/model.json";
const REAL_CUGAN_TILE_SIZE = 64;
const REAL_CUGAN_SCALE = 2;

let ortConfigured = false;
const ortSessionCache = new Map();
let realCuganModelPromise = null;

function configureOrt() {
  if (ortConfigured) return;
  const origin = self.location.origin;
  ort.env.wasm.wasmPaths = Object.fromEntries(
    Object.entries(ORT_WASM_PATHS).map(([fileName, path]) => [fileName, new URL(path, origin).href]),
  );
  ort.env.wasm.numThreads = 1;
  ortConfigured = true;
}

async function getOrtSession(upscaler) {
  configureOrt();
  const cached = ortSessionCache.get(upscaler.path);
  if (cached) return cached;

  const sessionPromise = ort.InferenceSession.create(upscaler.path, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  ortSessionCache.set(upscaler.path, sessionPromise);
  return sessionPromise;
}

function f32ToF16(value) {
  const floatView = new Float32Array(1);
  const intView = new Uint32Array(floatView.buffer);
  floatView[0] = value;
  const bits = intView[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = (bits >>> 23) & 0xff;
  let mantissa = bits & 0x7fffff;

  if (exponent === 0xff) {
    return sign | (mantissa ? 0x7e00 : 0x7c00);
  }

  exponent = exponent - 127 + 15;
  if (exponent >= 0x1f) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return sign | ((mantissa + 0x1000) >> 13);
  }

  return sign | (exponent << 10) | ((mantissa + 0x1000) >> 13);
}

function f16ToF32(value) {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const mantissa = value & 0x03ff;

  if (exponent === 0) {
    return sign * Math.pow(2, -14) * (mantissa / 1024);
  }
  if (exponent === 0x1f) {
    return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  }
  return sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

function tensorFloatValue(data, index) {
  const NativeFloat16Array = globalThis.Float16Array;
  if (NativeFloat16Array && data instanceof NativeFloat16Array) return data[index];
  if (data instanceof Float32Array || data instanceof Float64Array) return data[index];
  return f16ToF32(data[index]);
}

function clampByte(value) {
  return Math.max(0, Math.min(255, Math.round(value * 255)));
}

function alphaAtNearest(source, x, y, outputWidth, outputHeight) {
  const sourceX = Math.min(source.width - 1, Math.floor((x / outputWidth) * source.width));
  const sourceY = Math.min(source.height - 1, Math.floor((y / outputHeight) * source.height));
  return source.data[(sourceY * source.width + sourceX) * 4 + 3];
}

function splitRgbChannels(imageData) {
  const { data, width, height } = imageData;
  const pixelCount = width * height;
  const channels = [new Float32Array(pixelCount), new Float32Array(pixelCount), new Float32Array(pixelCount)];
  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    channels[0][index] = data[offset] / 255;
    channels[1][index] = data[offset + 1] / 255;
    channels[2][index] = data[offset + 2] / 255;
  }
  return channels;
}

function packRgbF16(imageData) {
  const { data, width, height } = imageData;
  const pixelCount = width * height;
  const NativeFloat16Array = globalThis.Float16Array;
  const tensor = NativeFloat16Array ? new NativeFloat16Array(pixelCount * 3) : new Uint16Array(pixelCount * 3);
  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    const red = data[offset] / 255;
    const green = data[offset + 1] / 255;
    const blue = data[offset + 2] / 255;
    if (NativeFloat16Array) {
      tensor[index] = red;
      tensor[pixelCount + index] = green;
      tensor[pixelCount * 2 + index] = blue;
    } else {
      tensor[index] = f32ToF16(red);
      tensor[pixelCount + index] = f32ToF16(green);
      tensor[pixelCount * 2 + index] = f32ToF16(blue);
    }
  }
  return tensor;
}

function imageDataFromMessage(message) {
  return {
    width: message.width,
    height: message.height,
    data: new Uint8ClampedArray(message.data),
  };
}

function writeRgbChannels(source, channels, width, height) {
  const data = new Uint8ClampedArray(width * height * 4);
  const pixelCount = width * height;
  for (let index = 0; index < pixelCount; index += 1) {
    const x = index % width;
    const y = Math.floor(index / width);
    const offset = index * 4;
    data[offset] = clampByte(channels[0][index]);
    data[offset + 1] = clampByte(channels[1][index]);
    data[offset + 2] = clampByte(channels[2][index]);
    data[offset + 3] = alphaAtNearest(source, x, y, width, height);
  }
  return data;
}

async function runArtCnn(message) {
  const upscaler = ORT_UPSCALERS[message.mode];
  if (!upscaler) throw new Error(`Unknown ArtCNN model: ${message.mode}`);

  const imageData = imageDataFromMessage(message);
  const session = await getOrtSession(upscaler);
  const channels = splitRgbChannels(imageData);
  const outputChannels = [];
  for (const channel of channels) {
    const input = new ort.Tensor("float32", channel, [1, 1, imageData.height, imageData.width]);
    const result = await session.run({ input });
    const output = result[session.outputNames[0]];
    outputChannels.push(output.data);
  }
  const outputWidth = imageData.width * 2;
  const outputHeight = outputChannels[0].length / outputWidth;
  return {
    width: outputWidth,
    height: outputHeight,
    data: writeRgbChannels(imageData, outputChannels, outputWidth, outputHeight),
  };
}

async function runAnimeJaNai(message) {
  const imageData = imageDataFromMessage(message);
  const session = await getOrtSession(ORT_UPSCALERS.animejanai);
  const input = new ort.Tensor("float16", packRgbF16(imageData), [1, 3, imageData.height, imageData.width]);
  const result = await session.run({ input });
  const output = result[session.outputNames[0]];
  const [batch, channels, outputHeight, outputWidth] = output.dims;
  if (batch !== 1 || channels !== 3) {
    throw new Error(`Unexpected AnimeJaNai output shape: ${output.dims.join("x")}`);
  }
  const channelSize = outputWidth * outputHeight;
  const rgb = [new Float32Array(channelSize), new Float32Array(channelSize), new Float32Array(channelSize)];
  for (let channel = 0; channel < 3; channel += 1) {
    const channelOffset = channel * channelSize;
    for (let index = 0; index < channelSize; index += 1) {
      rgb[channel][index] = tensorFloatValue(output.data, channelOffset + index);
    }
  }
  return {
    width: outputWidth,
    height: outputHeight,
    data: writeRgbChannels(imageData, rgb, outputWidth, outputHeight),
  };
}

async function getRealCuganModel() {
  if (!realCuganModelPromise) {
    realCuganModelPromise = Promise.all([
      import("@tensorflow/tfjs"),
      import("@tensorflow/tfjs-backend-webgpu").catch(() => null),
      import("@tensorflow/tfjs-backend-webgl").catch(() => null),
    ]).then(async ([tf]) => {
      if (self.navigator.gpu) {
        await tf.setBackend("webgpu").catch(() => tf.setBackend("webgl"));
      } else {
        await tf.setBackend("webgl");
      }
      await tf.ready();
      return { tf, model: await tf.loadGraphModel(REAL_CUGAN_MODEL_URL) };
    });
  }
  return realCuganModelPromise;
}

function realCuganTileInput(imageData, tileX, tileY) {
  const rgb = new Float32Array(REAL_CUGAN_TILE_SIZE * REAL_CUGAN_TILE_SIZE * 3);
  for (let y = 0; y < REAL_CUGAN_TILE_SIZE; y += 1) {
    const sourceY = Math.min(imageData.height - 1, tileY + y);
    for (let x = 0; x < REAL_CUGAN_TILE_SIZE; x += 1) {
      const sourceX = Math.min(imageData.width - 1, tileX + x);
      const sourceOffset = (sourceY * imageData.width + sourceX) * 4;
      const targetOffset = (y * REAL_CUGAN_TILE_SIZE + x) * 3;
      rgb[targetOffset] = imageData.data[sourceOffset] / 255;
      rgb[targetOffset + 1] = imageData.data[sourceOffset + 1] / 255;
      rgb[targetOffset + 2] = imageData.data[sourceOffset + 2] / 255;
    }
  }
  return rgb;
}

async function runRealCuganTile(tf, model, imageData, tileX, tileY) {
  const outputTensor = tf.tidy(() => {
    const input = tf.tensor4d(realCuganTileInput(imageData, tileX, tileY), [
      1,
      REAL_CUGAN_TILE_SIZE,
      REAL_CUGAN_TILE_SIZE,
      3,
    ]);
    const result = model.execute(input);
    const image = Array.isArray(result) ? result[0] : result;
    return image.squeeze().clipByValue(0, 1);
  });

  try {
    const [outputHeight, outputWidth, channels] = outputTensor.shape;
    if (outputWidth !== REAL_CUGAN_TILE_SIZE * REAL_CUGAN_SCALE || outputHeight !== REAL_CUGAN_TILE_SIZE * REAL_CUGAN_SCALE || channels !== 3) {
      throw new Error(`Unexpected Real-CUGAN tile output shape: ${outputTensor.shape.join("x")}`);
    }
    return await outputTensor.data();
  } finally {
    outputTensor.dispose();
  }
}

async function runRealCugan(message) {
  const { tf, model } = await getRealCuganModel();
  const imageData = imageDataFromMessage(message);
  const outputWidth = imageData.width * REAL_CUGAN_SCALE;
  const outputHeight = imageData.height * REAL_CUGAN_SCALE;
  const data = new Uint8ClampedArray(outputWidth * outputHeight * 4);

  for (let tileY = 0; tileY < imageData.height; tileY += REAL_CUGAN_TILE_SIZE) {
    for (let tileX = 0; tileX < imageData.width; tileX += REAL_CUGAN_TILE_SIZE) {
      const values = await runRealCuganTile(tf, model, imageData, tileX, tileY);
      const validWidth = Math.min(REAL_CUGAN_TILE_SIZE, imageData.width - tileX) * REAL_CUGAN_SCALE;
      const validHeight = Math.min(REAL_CUGAN_TILE_SIZE, imageData.height - tileY) * REAL_CUGAN_SCALE;
      for (let y = 0; y < validHeight; y += 1) {
        for (let x = 0; x < validWidth; x += 1) {
          const tileOffset = (y * REAL_CUGAN_TILE_SIZE * REAL_CUGAN_SCALE + x) * 3;
          const outputX = tileX * REAL_CUGAN_SCALE + x;
          const outputY = tileY * REAL_CUGAN_SCALE + y;
          const targetOffset = (outputY * outputWidth + outputX) * 4;
          data[targetOffset] = clampByte(values[tileOffset]);
          data[targetOffset + 1] = clampByte(values[tileOffset + 1]);
          data[targetOffset + 2] = clampByte(values[tileOffset + 2]);
          data[targetOffset + 3] = alphaAtNearest(imageData, outputX, outputY, outputWidth, outputHeight);
        }
      }
    }
  }

  return { width: outputWidth, height: outputHeight, data };
}

self.addEventListener("message", (event) => {
  const message = event.data;
  Promise.resolve()
    .then(() => {
      if (message.mode === "animejanai") return runAnimeJaNai(message);
      if (message.mode === "realcugan") return runRealCugan(message);
      return runArtCnn(message);
    })
    .then((result) => {
      self.postMessage(
        {
          id: message.id,
          ok: true,
          width: result.width,
          height: result.height,
          data: result.data.buffer,
        },
        [result.data.buffer],
      );
    })
    .catch((error) => {
      self.postMessage({
        id: message.id,
        ok: false,
        error: error instanceof Error ? error.message : "Upscaler worker failed",
      });
    });
});
