// this file will be hot reloaded often

#if __INTELLISENSE__
#include <moonlightsdk.h>
#include <stereokit.h>
#include <stereokit_ui.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#endif 

using namespace sk;

bool initialize_stereokit() {
  sk_settings_t settings = {};
  settings.android_java_vm = java_vm;
  settings.android_activity = (void *)java_android_activity; 

  settings.app_name           = "SunflowerOS cppyy v0.1";
  settings.assets_folder      = "/data/data/com.termux/files/home/MagicLeap2-Synced/StereoKit/Examples/Assets/";
  settings.display_preference = display_mode_mixedreality;

  backend_openxr_ext_request("XR_ML_ml2_controller_interaction");
  printf("Initializing SK");
  if (!sk_init(settings)) {
    printf("SK Init failed");
    return false;
  }
  return true;
}


// on sigint, k

pose_t cube_pose;
mesh_t cube_mesh;
material_t cube_mat;

mesh_t video_mesh;
material_t video_mat;
tex_t video_tex;

model_t avatar;
AVFrame* latestFrameRGBA;
// sws_ctx
struct SwsContext *sws_ctx;

void on_moonlight_frame(AVFrame *frame) {
  // think this happens in a different thread
  if (sws_ctx == nullptr) {
    sws_ctx = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format, frame->width, frame->height, AV_PIX_FMT_RGBA, SWS_BILINEAR, NULL, NULL, NULL);
  }
  if (latestFrameRGBA == nullptr) {
    latestFrameRGBA = av_frame_alloc();
    latestFrameRGBA->width = frame->width;
    latestFrameRGBA->height = frame->height;
    latestFrameRGBA->format = AV_PIX_FMT_RGBA;
    av_frame_get_buffer(latestFrameRGBA, 0);
  }
  sws_scale(sws_ctx, (uint8_t const* const*)frame->data, frame->linesize, 0, frame->height, latestFrameRGBA->data, latestFrameRGBA->linesize);

  // idk why, but if have this here it crashes jupyer notebook when try to cleanup, w/o error
  // tex_set_colors(video_tex, latestFrameRGBA->width, latestFrameRGBA->height, latestFrameRGBA->data[0]);
}

void setup() {

  cube_pose = {{0,0,-0.5f}, quat_identity};
  cube_mesh = mesh_gen_rounded_cube(vec3_one * 1 * 1, 0.02f, 4);
  cube_mat  = material_find(default_id_material);

  video_mesh = mesh_gen_plane(vec2{1920/1080.0, 1} * 0.5f, { 0,0,-1 }, {0,1,0});
  video_mat = material_find(default_id_material_unlit);
  video_tex = tex_create(tex_type_image, tex_format_rgba32);
  material_set_texture(video_mat, "diffuse", video_tex);

  avatar = model_create_file("/data/data/com.termux/files/home/MagicLeap2-Synced/vroiddemo_throughblender.glb");

  // moon_init_config("192.168.0.4", "Desktop");
  // moon_init_server_connection();
  // moon_start_stream(on_moonlight_frame);
}

void update_fn() {
  // render_add_mesh(cube_mesh, cube_mat, pose_matrix(cube_pose, vec3_one));
  // model_draw(avatar,  pose_matrix({{0, 0, 0}, quat_identity}, vec3_one * 1.0));

	float    size   = tex_get_height(video_tex) * 0.0004f;
	bounds_t bounds = mesh_get_bounds(video_mesh);
	bounds.center     = (bounds.center + vec3{ 0, -bounds.dimensions.y / 2, bounds.dimensions.z / 2 }) * size + vec3{0,-0.04f,-0.02f};
	bounds.dimensions = vec3{0.2f, 0.02f, 0.02f};

	// UI for the grab handle

	static pose_t desktop_pose        = pose_t{ vec3{0,0,-0.5f}, quat_lookat({0,0,-1}, input_head()->position)};
	ui_enable_far_interact(false);
	ui_handle_begin("Desktop", desktop_pose, bounds, true);
	ui_handle_end();
	ui_enable_far_interact(true);

  render_add_mesh(video_mesh, video_mat, pose_matrix(desktop_pose, vec3_one * 1.0));
  // render_add_mesh(video_mesh, video_mat, pose_matrix(desktop_pose, vec3_one * 0.2));

  if (latestFrameRGBA != nullptr) {
    tex_set_colors(video_tex, latestFrameRGBA->width, latestFrameRGBA->height, latestFrameRGBA->data[0]);
  }
}

void update() {
  sk_step(update_fn);
}

void cleanup() {
  // moon_stop_stream();
  sws_freeContext(sws_ctx);
  av_frame_free(&latestFrameRGBA);

  mesh_release(cube_mesh);
  material_release(cube_mat);
  model_release(avatar);

  material_release(video_mat);
  tex_release(video_tex);
  mesh_release(video_mesh);

}