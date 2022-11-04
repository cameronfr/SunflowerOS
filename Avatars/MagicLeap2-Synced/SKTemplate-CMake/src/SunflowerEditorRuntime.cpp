#include <zmq.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ctime>

#include <map>
#include <thread>

void *zmq_context;
void *zmq_pub_sock; //sends data to VSCode
void *zmq_gather_sock; //gathers from each thread's  sockets
// map of thread id to its socket
std::map<std::thread::id, void *> zmq_sock_map;
std::map<std::thread::id, std::mutex> zmq_mutex_map;
std::mutex zmq_maps_lock;
bool zmq_stopped;

const int LINGER_ZERO = 0;

std::thread heartbeat_thread;
bool heartbeat_thread_running;
std::thread proxy_thread;

// These are called from non-main threads
int hearbeatnum = 0;
void msgserver_inthread_send(const char* topic, const char* msg) {
  std::thread::id tid = std::this_thread::get_id();
  // if mutex is not in the map, we need to lock. since the [] op will create it if it doesn't exist. 
  if (zmq_mutex_map.count(tid) == 0) {
    std::lock_guard<std::mutex> lock(zmq_maps_lock);
  }
  std::lock_guard<std::mutex> lock(zmq_mutex_map[tid]);
  if (zmq_stopped) {
    printf("msgserver_inthread_send: zmq_stopped but thread trying to send");
    return;
  }
  // printf("msgserver_inthread_send start: %ld", tid);
  void *zmq_sock = zmq_sock_map[tid];
  if (zmq_sock == NULL) {
    printf("Creating new zmq socket for thread %ld",tid);
    zmq_sock = zmq_socket(zmq_context, ZMQ_PUB);
    int rc = zmq_connect(zmq_sock, "inproc://gather");
    if (rc == -1) {
      printf("Error connecting to gather socket: %s", zmq_strerror(errno));
    }
    zmq_sock_map[tid] = zmq_sock;
    // TODO: first msg from a thread is always lost
    // hack solution: sleep for 20ms
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  zmq_send(zmq_sock, topic, strlen(topic), ZMQ_SNDMORE);
  zmq_send(zmq_sock, msg, strlen(msg), 0);
  // printf("msgserver_inthread_send end: %ld", tid);
}
void msgserver_inthread_sendlog(char *filepath, int line, int lineChar, char *text) {
  const static char *formatString = R"({"filepath":"%s","line":%d,"lineChar":%d,"text":"%s","timestamp":%ld})";

  long timestamp = std::time(0);
  int bufSize = snprintf(NULL, 0, formatString, filepath, line, lineChar, text, timestamp);
  char *msg = (char *)malloc(bufSize + 1);
  snprintf(msg, bufSize + 1, formatString, filepath, line, lineChar, text, timestamp);

  msgserver_inthread_send("log", msg);
  free(msg);
}
// These are called from main thread
const char* msgserver_init() {
  zmq_context = zmq_ctx_new();
  zmq_stopped = false;

  zmq_pub_sock = zmq_socket(zmq_context, ZMQ_XPUB);
  int timeout = 2000;
  zmq_setsockopt(zmq_pub_sock, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
  zmq_setsockopt(zmq_pub_sock, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
  int rc = zmq_bind(zmq_pub_sock, "tcp://0.0.0.0:5895");
  if (rc == -1) {
    const char* error = zmq_strerror(zmq_errno());
    printf("Error binding pub socket: %s", error);
    return error; 
  }

  zmq_gather_sock = zmq_socket(zmq_context, ZMQ_XSUB);
  rc = zmq_bind(zmq_gather_sock, "inproc://gather");
  if (rc == -1) {
    const char* error = zmq_strerror(zmq_errno());
    printf("Error binding gather socket: %s", error);
    return error; 
  }

  printf("Starting proxy thread\n");
  proxy_thread = std::thread([]() {
    // note that we're passing sockets to this thread, no touching them from main thread again
    // note that socks must be XPUB and XSUB (not non-x version), otherwise doesn't work
    printf("Starting proxy\n");
    zmq_proxy(zmq_gather_sock, zmq_pub_sock, NULL);
    printf("Proxy thread ending");
    // zmq_proxy exits when zmq_ctx_term is called, we must close the sockets to unblock it
    zmq_setsockopt(zmq_pub_sock, ZMQ_LINGER, &LINGER_ZERO, sizeof(LINGER_ZERO));
    zmq_setsockopt(zmq_gather_sock, ZMQ_LINGER, &LINGER_ZERO, sizeof(LINGER_ZERO));
    zmq_close(zmq_pub_sock);
    zmq_close(zmq_gather_sock);
    printf("Proxy thread finished closing sockets\n");
  });

  printf("Starting heartbeat thread\n");
  heartbeat_thread_running = true;
  heartbeat_thread = std::thread([]() {
    while (heartbeat_thread_running) {
      char* hearbeatNum = (char*)malloc(10);
      sprintf(hearbeatNum, "%d", hearbeatnum++);
      msgserver_inthread_send("heartbeat",hearbeatNum);
      free(hearbeatNum);
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  });

  return "Success";
}
void msgserver_close() {
  // Stop heartbeat thread
  heartbeat_thread_running = false;
  heartbeat_thread.join();
  // Close sockets
  zmq_stopped = true;
  std::lock_guard<std::mutex> lock(zmq_maps_lock);
  for (auto it = zmq_mutex_map.begin(); it != zmq_mutex_map.end(); ++it) {
    std::thread::id tid = it->first;
    std::lock_guard<std::mutex> lock(zmq_mutex_map[tid]); 
    void *zmq_sock = zmq_sock_map[tid];
    zmq_setsockopt(zmq_sock, ZMQ_LINGER, &LINGER_ZERO, sizeof(LINGER_ZERO));
    zmq_close(zmq_sock);
  }
  zmq_sock_map.clear();
  zmq_mutex_map.clear();
  printf("Waiting for proxy thread to end\n");
  zmq_ctx_shutdown(zmq_context); 
  printf("Context shutdown");
  proxy_thread.join(); //zmq_proxy stops when context is terminated
  printf("Proxy thread joined successfully\n");
  zmq_ctx_term(zmq_context);
  printf("Context terminated");
  zmq_context = NULL;
}

// ------------------------------

void amIHere() {
  printf("I'm here! %d %s\n", 23+12, "hello"  "1");
}
