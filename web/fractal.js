(function (global) {
  'use strict';

  const IMAGE_SIZE = 512;
  const RANGE_SIZE = 4;
  const DOMAIN_SIZE = 8;
  const SYMMETRY_COUNT = 8;

  const RANGE_AREA = RANGE_SIZE * RANGE_SIZE;
  const DOMAIN_AREA = DOMAIN_SIZE * DOMAIN_SIZE;
  const RANGE_BLOCKS_PER_SIDE = IMAGE_SIZE / RANGE_SIZE;
  const DOMAIN_BLOCKS_PER_SIDE = IMAGE_SIZE / DOMAIN_SIZE;
  const NUM_RANGE_BLOCKS = RANGE_BLOCKS_PER_SIDE * RANGE_BLOCKS_PER_SIDE;
  const NUM_DOMAIN_BLOCKS = DOMAIN_BLOCKS_PER_SIDE * DOMAIN_BLOCKS_PER_SIDE;
  const VARIANT_COUNT = NUM_DOMAIN_BLOCKS * SYMMETRY_COUNT;

  function createSeededRandom(seed) {
    let state = (seed >>> 0) || 1;
    return function () {
      state = (state * 1664525 + 1013904223) >>> 0;
      return state / 0xffffffff;
    };
  }

  function toFloat64Array(bufferLike) {
    if (bufferLike instanceof Float64Array) {
      return bufferLike;
    }
    const arr = new Float64Array(bufferLike.length);
    for (let i = 0; i < bufferLike.length; i++) {
      arr[i] = bufferLike[i];
    }
    return arr;
  }

  function transformIndex(x, y, sym, size) {
    const rot = sym & 3;
    const flip = sym >= 4;
    const fx = flip ? size - 1 - x : x;
    let srcX;
    let srcY;
    switch (rot) {
      case 0:
        srcY = y;
        srcX = fx;
        break;
      case 1:
        srcY = fx;
        srcX = size - 1 - y;
        break;
      case 2:
        srcY = size - 1 - y;
        srcX = size - 1 - fx;
        break;
      case 3:
        srcY = size - 1 - fx;
        srcX = y;
        break;
      default:
        srcY = y;
        srcX = fx;
    }
    return srcY * size + srcX;
  }

  function transformBlock(src, srcOffset, sym, size, dest, destOffset) {
    for (let y = 0; y < size; y++) {
      const rowOffset = destOffset + y * size;
      for (let x = 0; x < size; x++) {
        const srcIdx = transformIndex(x, y, sym, size);
        dest[rowOffset + x] = src[srcOffset + srcIdx];
      }
    }
  }

  function downscaleBlock(src, srcOffset, size, dest, destOffset) {
    const half = size >> 1;
    for (let y = 0; y < half; y++) {
      const baseSrcY = (y << 1) * size + srcOffset;
      const baseDest = destOffset + y * half;
      for (let x = 0; x < half; x++) {
        const baseSrc = baseSrcY + (x << 1);
        const a = src[baseSrc];
        const b = src[baseSrc + 1];
        const c = src[baseSrc + size];
        const d = src[baseSrc + size + 1];
        dest[baseDest + x] = (a + b + c + d) * 0.25;
      }
    }
  }

  function extractDomainBlocks(pixels) {
    const result = new Float64Array(NUM_DOMAIN_BLOCKS * DOMAIN_AREA);
    let blockIndex = 0;
    for (let by = 0; by < DOMAIN_BLOCKS_PER_SIDE; by++) {
      for (let bx = 0; bx < DOMAIN_BLOCKS_PER_SIDE; bx++) {
        const destBase = blockIndex * DOMAIN_AREA;
        const srcBaseY = by * DOMAIN_SIZE * IMAGE_SIZE;
        const srcBaseX = bx * DOMAIN_SIZE;
        for (let y = 0; y < DOMAIN_SIZE; y++) {
          const srcRow = srcBaseY + y * IMAGE_SIZE + srcBaseX;
          const destRow = destBase + y * DOMAIN_SIZE;
          for (let x = 0; x < DOMAIN_SIZE; x++) {
            result[destRow + x] = pixels[srcRow + x];
          }
        }
        blockIndex++;
      }
    }
    return result;
  }

  function buildDomainVariants(domainBlocks) {
    const variants = new Float64Array(VARIANT_COUNT * RANGE_AREA);
    const centered = new Float64Array(VARIANT_COUNT * RANGE_AREA);
    const means = new Float64Array(VARIANT_COUNT);
    const energies = new Float64Array(VARIANT_COUNT);
    const transformed = new Float64Array(DOMAIN_AREA);
    const downscaled = new Float64Array(RANGE_AREA);

    for (let domainIdx = 0; domainIdx < NUM_DOMAIN_BLOCKS; domainIdx++) {
      const srcOffset = domainIdx * DOMAIN_AREA;
      for (let sym = 0; sym < SYMMETRY_COUNT; sym++) {
        const variantIdx = domainIdx * SYMMETRY_COUNT + sym;
        const varOffset = variantIdx * RANGE_AREA;

        transformBlock(domainBlocks, srcOffset, sym, DOMAIN_SIZE, transformed, 0);
        downscaleBlock(transformed, 0, DOMAIN_SIZE, downscaled, 0);

        let sum = 0;
        for (let i = 0; i < RANGE_AREA; i++) {
          const value = downscaled[i];
          variants[varOffset + i] = value;
          sum += value;
        }
        const mean = sum / RANGE_AREA;
        means[variantIdx] = mean;

        let energy = 0;
        for (let i = 0; i < RANGE_AREA; i++) {
          const diff = variants[varOffset + i] - mean;
          centered[varOffset + i] = diff;
          energy += diff * diff;
        }
        energies[variantIdx] = energy;
      }
    }

    return { variants, centered, means, energies };
  }

  function compressFromPixels(pixelSource) {
    const pixels = toFloat64Array(pixelSource);
    if (pixels.length !== IMAGE_SIZE * IMAGE_SIZE) {
      throw new Error('Expected flat grayscale image of size 512x512.');
    }

    const domainBlocks = extractDomainBlocks(pixels);
    const domainData = buildDomainVariants(domainBlocks);

    const domainIndices = new Int32Array(NUM_RANGE_BLOCKS);
    const symmetries = new Uint8Array(NUM_RANGE_BLOCKS);
    const contrasts = new Float32Array(NUM_RANGE_BLOCKS);
    const offsets = new Float32Array(NUM_RANGE_BLOCKS);

    const { variants, centered, means, energies } = domainData;

    const target = new Float64Array(RANGE_AREA);
    const targetCentered = new Float64Array(RANGE_AREA);

    const rangeBlocksPerSide = RANGE_BLOCKS_PER_SIDE;
    const variantCount = VARIANT_COUNT;

    for (let blockIdx = 0; blockIdx < NUM_RANGE_BLOCKS; blockIdx++) {
      const ry = Math.floor(blockIdx / rangeBlocksPerSide) * RANGE_SIZE;
      const rx = (blockIdx % rangeBlocksPerSide) * RANGE_SIZE;

      let sum = 0;
      for (let dy = 0; dy < RANGE_SIZE; dy++) {
        const srcRow = (ry + dy) * IMAGE_SIZE + rx;
        const targetRow = dy * RANGE_SIZE;
        for (let dx = 0; dx < RANGE_SIZE; dx++) {
          const value = pixels[srcRow + dx];
          target[targetRow + dx] = value;
          sum += value;
        }
      }

      const mean = sum / RANGE_AREA;
      let targetEnergy = 0;
      for (let i = 0; i < RANGE_AREA; i++) {
        const diff = target[i] - mean;
        targetCentered[i] = diff;
        targetEnergy += diff * diff;
      }

      let bestError = Infinity;
      let bestVariant = 0;
      let bestContrast = 0;

      for (let variantIdx = 0; variantIdx < variantCount; variantIdx++) {
        const energy = energies[variantIdx];
        const varOffset = variantIdx * RANGE_AREA;
        let numerator = 0;
        for (let i = 0; i < RANGE_AREA; i++) {
          numerator += centered[varOffset + i] * targetCentered[i];
        }

        let contrast;
        let error;
        if (energy > 1e-6) {
          contrast = numerator / energy;
          error = targetEnergy - (numerator * numerator) / energy;
        } else {
          contrast = 0;
          error = targetEnergy;
        }

        if (error < bestError) {
          bestError = error;
          bestVariant = variantIdx;
          bestContrast = contrast;
        }
      }

      const domainIndex = Math.floor(bestVariant / SYMMETRY_COUNT);
      const symmetry = bestVariant % SYMMETRY_COUNT;
      const meanDomain = means[bestVariant];
      const offset = mean - bestContrast * meanDomain;

      domainIndices[blockIdx] = domainIndex;
      symmetries[blockIdx] = symmetry;
      contrasts[blockIdx] = bestContrast;
      offsets[blockIdx] = offset;
    }

    return {
      width: IMAGE_SIZE,
      height: IMAGE_SIZE,
      rangeSize: RANGE_SIZE,
      domainSize: DOMAIN_SIZE,
      symmetryCount: SYMMETRY_COUNT,
      domain: domainIndices,
      symmetry: symmetries,
      contrast: contrasts,
      offset: offsets,
    };
  }

  function iterate(image, pairs) {
    const next = new Float64Array(image.length);
    const domainTemp = new Float64Array(DOMAIN_AREA);
    const transformed = new Float64Array(DOMAIN_AREA);
    const downscaled = new Float64Array(RANGE_AREA);

    for (let blockIdx = 0; blockIdx < NUM_RANGE_BLOCKS; blockIdx++) {
      const domainIndex = pairs.domain[blockIdx];
      const sym = pairs.symmetry[blockIdx];
      const contrast = pairs.contrast[blockIdx];
      const offset = pairs.offset[blockIdx];

      const rangeY = Math.floor(blockIdx / RANGE_BLOCKS_PER_SIDE) * RANGE_SIZE;
      const rangeX = (blockIdx % RANGE_BLOCKS_PER_SIDE) * RANGE_SIZE;

      const domainY = Math.floor(domainIndex / DOMAIN_BLOCKS_PER_SIDE) * DOMAIN_SIZE;
      const domainX = (domainIndex % DOMAIN_BLOCKS_PER_SIDE) * DOMAIN_SIZE;

      for (let dy = 0; dy < DOMAIN_SIZE; dy++) {
        const srcRow = (domainY + dy) * IMAGE_SIZE + domainX;
        const destRow = dy * DOMAIN_SIZE;
        for (let dx = 0; dx < DOMAIN_SIZE; dx++) {
          domainTemp[destRow + dx] = image[srcRow + dx];
        }
      }

      transformBlock(domainTemp, 0, sym, DOMAIN_SIZE, transformed, 0);
      downscaleBlock(transformed, 0, DOMAIN_SIZE, downscaled, 0);

      for (let ry = 0; ry < RANGE_SIZE; ry++) {
        const destRow = (rangeY + ry) * IMAGE_SIZE + rangeX;
        const srcRow = ry * RANGE_SIZE;
        for (let rx = 0; rx < RANGE_SIZE; rx++) {
          next[destRow + rx] = contrast * downscaled[srcRow + rx] + offset;
        }
      }
    }

    return next;
  }

  function decompress(pairs, iterations = 8, seed = 1234) {
    const rng = createSeededRandom(seed);
    const image = new Float64Array(IMAGE_SIZE * IMAGE_SIZE);
    for (let i = 0; i < image.length; i++) {
      image[i] = rng() * 255;
    }

    let current = image;
    for (let i = 0; i < iterations; i++) {
      current = iterate(current, pairs);
    }
    return current;
  }

  function pairsFromObject(obj) {
    return {
      domain: new Int32Array(obj.domain),
      symmetry: new Uint8Array(obj.symmetry || obj.sym),
      contrast: new Float32Array(obj.contrast),
      offset: new Float32Array(obj.offset),
    };
  }

  function pairsToObject(pairs) {
    return {
      domain: Array.from(pairs.domain),
      symmetry: Array.from(pairs.symmetry),
      contrast: Array.from(pairs.contrast),
      offset: Array.from(pairs.offset),
    };
  }

  const api = {
    IMAGE_SIZE,
    RANGE_SIZE,
    DOMAIN_SIZE,
    SYMMETRY_COUNT,
    compressFromPixels,
    iterate,
    decompress,
    pairsFromObject,
    pairsToObject,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else {
    global.FractalCodec = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
