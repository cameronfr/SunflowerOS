
#include <libavcodec/avcodec.h>
#include <stdbool.h>

extern bool moon_init_config(char* address, char* app);
extern bool moon_init_server_connection();
extern bool moon_pair_server();
extern bool moon_start_stream(/*void (*on_frame)(AVFrame*)*/void *on_frame);
extern void moon_stop_stream(void);