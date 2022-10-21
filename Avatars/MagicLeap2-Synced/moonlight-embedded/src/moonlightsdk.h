
#include <libavcodec/avcodec.h>

extern bool init_config();
extern bool init_server_connection();
extern bool pair_server();
extern bool start_stream(void (*on_frame)(AVFrame*));
extern void stop_stream(void);