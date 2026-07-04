// SPDX-Licence-Identifier: EUPL-1.2

package compute

type kernelSpec struct {
	inputNames  []string
	outputNames []string
	source      string
}

var computeKernelSpecs = map[string]kernelSpec{
	"frame_copy_scale": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint dst_x = thread_position_in_grid.x;
uint dst_y = thread_position_in_grid.y;
if (dst_x >= DST_WIDTH || dst_y >= DST_HEIGHT) {
    return;
}
uint src_x = (dst_x * SRC_WIDTH) / DST_WIDTH;
uint src_y = (dst_y * SRC_HEIGHT) / DST_HEIGHT;
uint src_index = src_y * SRC_STRIDE + src_x * BPP;
uint dst_index = dst_y * DST_STRIDE + dst_x * BPP;
for (int channel = 0; channel < BPP; channel++) {
    dst[dst_index + channel] = src[src_index + channel];
}`,
	},
	"frame_bilinear_rgba": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint dst_x = thread_position_in_grid.x;
uint dst_y = thread_position_in_grid.y;
if (dst_x >= DST_WIDTH || dst_y >= DST_HEIGHT) {
    return;
}
float src_x = ((float(dst_x) + 0.5f) * float(SRC_WIDTH) / float(DST_WIDTH)) - 0.5f;
float src_y = ((float(dst_y) + 0.5f) * float(SRC_HEIGHT) / float(DST_HEIGHT)) - 0.5f;
int x0 = int(metal::floor(src_x));
int y0 = int(metal::floor(src_y));
float tx = src_x - float(x0);
float ty = src_y - float(y0);
x0 = metal::clamp(x0, 0, SRC_WIDTH - 1);
y0 = metal::clamp(y0, 0, SRC_HEIGHT - 1);
int x1 = metal::clamp(x0 + 1, 0, SRC_WIDTH - 1);
int y1 = metal::clamp(y0 + 1, 0, SRC_HEIGHT - 1);
uint dst_index = dst_y * DST_STRIDE + dst_x * 4;
uint tl = uint(y0) * SRC_STRIDE + uint(x0) * 4;
uint tr = uint(y0) * SRC_STRIDE + uint(x1) * 4;
uint bl = uint(y1) * SRC_STRIDE + uint(x0) * 4;
uint br = uint(y1) * SRC_STRIDE + uint(x1) * 4;
for (int channel = 0; channel < 4; channel++) {
    float top = float(src[tl + uint(channel)]) + (float(src[tr + uint(channel)]) - float(src[tl + uint(channel)])) * tx;
    float bottom = float(src[bl + uint(channel)]) + (float(src[br + uint(channel)]) - float(src[bl + uint(channel)])) * tx;
    float value = top + (bottom - top) * ty;
    dst[dst_index + uint(channel)] = uchar(metal::clamp(metal::rint(value), 0.0f, 255.0f));
}`,
	},
	"frame_rgb565_to_rgba8": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint src_index = y * SRC_STRIDE + x * 2;
ushort packed = ushort(src[src_index]) | (ushort(src[src_index + 1]) << 8);
uchar r = uchar((((packed >> 11) & 0x1F) * 255 + 15) / 31);
uchar g = uchar((((packed >> 5) & 0x3F) * 255 + 31) / 63);
uchar b = uchar(((packed & 0x1F) * 255 + 15) / 31);
uint dst_index = y * DST_STRIDE + x * 4;
dst[dst_index + 0] = r;
dst[dst_index + 1] = g;
dst[dst_index + 2] = b;
dst[dst_index + 3] = 255;`,
	},
	"frame_channel_swizzle": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint src_index = y * SRC_STRIDE + x * 4;
uint dst_index = y * DST_STRIDE + x * 4;
dst[dst_index + 0] = src[src_index + 2];
dst[dst_index + 1] = src[src_index + 1];
dst[dst_index + 2] = src[src_index + 0];
dst[dst_index + 3] = src[src_index + 3];`,
	},
	"frame_xrgb8888_to_rgba8": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint src_index = y * SRC_STRIDE + x * 4;
uint dst_index = y * DST_STRIDE + x * 4;
uchar b = src[src_index + 0];
uchar g = src[src_index + 1];
uchar r = src[src_index + 2];
dst[dst_index + 0] = r;
dst[dst_index + 1] = g;
dst[dst_index + 2] = b;
dst[dst_index + 3] = 255;`,
	},
	"frame_palette_expand_rgba8": {
		inputNames:  []string{"src", "palette"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint src_index = y * SRC_STRIDE + x;
uint palette_index = uint(src[src_index]) * 4;
uint dst_index = y * DST_STRIDE + x * 4;
dst[dst_index + 0] = palette[palette_index + 0];
dst[dst_index + 1] = palette[palette_index + 1];
dst[dst_index + 2] = palette[palette_index + 2];
dst[dst_index + 3] = palette[palette_index + 3];`,
	},
	"frame_scanline_filter": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint index = y * STRIDE + x * 4;
float scan = ((y & 1u) == 0u) ? 1.0f : (1.0f - float(STRENGTH) / 256.0f);
for (uint channel = 0; channel < 3; channel++) {
    float value = float(src[index + channel]) * scan;
    dst[index + channel] = uchar(metal::clamp(metal::rint(value), 0.0f, 255.0f));
}
dst[index + 3] = src[index + 3];`,
	},
	"frame_crt_filter": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint index = y * STRIDE + x * 4;
uint r_index = BGRA_ORDER ? 2u : 0u;
uint g_index = 1u;
uint b_index = BGRA_ORDER ? 0u : 2u;
float scan = ((y & 1u) == 0u) ? 1.0f : (1.0f - float(SCANLINE_STRENGTH) / 256.0f);
float shadow = 1.0f - float(MASK_STRENGTH) / 256.0f;
float r_mask = shadow;
float g_mask = shadow;
float b_mask = shadow;
switch (x % 3u) {
case 0u:
    r_mask = 1.0f;
    break;
case 1u:
    g_mask = 1.0f;
    break;
default:
    b_mask = 1.0f;
    break;
}
float r = float(src[index + r_index]) * scan * r_mask;
float g = float(src[index + g_index]) * scan * g_mask;
float b = float(src[index + b_index]) * scan * b_mask;
dst[index + r_index] = uchar(metal::clamp(metal::rint(r), 0.0f, 255.0f));
dst[index + g_index] = uchar(metal::clamp(metal::rint(g), 0.0f, 255.0f));
dst[index + b_index] = uchar(metal::clamp(metal::rint(b), 0.0f, 255.0f));
dst[index + 3] = src[index + 3];`,
	},
	"frame_soften_filter": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint index = y * STRIDE + x * 4;
float mix = float(STRENGTH) / 256.0f;
for (uint channel = 0; channel < 3; channel++) {
    float sum = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        int sy = metal::clamp(int(y) + dy, 0, HEIGHT - 1);
        for (int dx = -1; dx <= 1; dx++) {
            int sx = metal::clamp(int(x) + dx, 0, WIDTH - 1);
            uint sample_index = uint(sy) * STRIDE + uint(sx) * 4 + channel;
            sum += float(src[sample_index]);
        }
    }
    float blurred = sum / 9.0f;
    float original = float(src[index + channel]);
    float value = original + (blurred - original) * mix;
    dst[index + channel] = uchar(metal::clamp(metal::rint(value), 0.0f, 255.0f));
}
dst[index + 3] = src[index + 3];`,
	},
	"frame_sharpen_filter": {
		inputNames:  []string{"src"},
		outputNames: []string{"dst"},
		source: `uint x = thread_position_in_grid.x;
uint y = thread_position_in_grid.y;
if (x >= WIDTH || y >= HEIGHT) {
    return;
}
uint index = y * STRIDE + x * 4;
float mix = float(STRENGTH) / 256.0f;
for (uint channel = 0; channel < 3; channel++) {
    float sum = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        int sy = metal::clamp(int(y) + dy, 0, HEIGHT - 1);
        for (int dx = -1; dx <= 1; dx++) {
            int sx = metal::clamp(int(x) + dx, 0, WIDTH - 1);
            uint sample_index = uint(sy) * STRIDE + uint(sx) * 4 + channel;
            sum += float(src[sample_index]);
        }
    }
    float blurred = sum / 9.0f;
    float original = float(src[index + channel]);
    float value = original + (original - blurred) * mix;
    dst[index + channel] = uchar(metal::clamp(metal::rint(value), 0.0f, 255.0f));
}
dst[index + 3] = src[index + 3];`,
	},
}

const computeKernelHeader = "#include <metal_stdlib>\nusing namespace metal;\n"
