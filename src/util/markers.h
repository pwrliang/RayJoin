
#ifndef UTIL_MARKERS_H
#define UTIL_MARKERS_H
#include <nvToolsExt.h>

namespace rayjoin {

// #define DISABLE_NVTX_MARKERS

/**
Example Usage:
{
    RangeMarker marker (true, "Scoped marker");

    {
        RangeMarker another;
        Marker::MarkDouble(1337.5);
        another.Stop();
        Marker::Mark();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
}
*/

static const int CATEGORY_KERNEL_WORKITEMS = 1;
static const int CATEGORY_INTERVAL_WORKITEMS = 2;

struct Marker {
  static nvtxEventAttributes_t CreateEvent(const char* message = "Marker",
                                           int color = 0, int category = 0) {
    nvtxEventAttributes_t eventAttrib = {0};
    // set the version and the size information
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    // configure the attributes.  0 is the default for all attributes.
    if (color) {
      eventAttrib.colorType = NVTX_COLOR_ARGB;
      eventAttrib.color = 0xFF880000;
    }
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message;
    eventAttrib.category = category;

    return eventAttrib;
  }

  static void Mark(const char* message = "Marker", int color = 0,
                   int category = 0) {
    nvtxEventAttributes_t ev = CreateEvent(message, color, category);
#ifndef DISABLE_NVTX_MARKERS
    nvtxMarkEx(&ev);
#endif
  }

  static void MarkInt(int64_t value, const char* message = "Marker",
                      int color = 0, int category = 0) {
    nvtxEventAttributes_t ev = CreateEvent(message, color, category);
    ev.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    ev.payload.llValue = value;
#ifndef DISABLE_NVTX_MARKERS
    nvtxMarkEx(&ev);
#endif
  }

  static void MarkUnsignedInt(uint64_t value, const char* message = "Marker",
                              int color = 0, int category = 0) {
    nvtxEventAttributes_t ev = CreateEvent(message, color, category);
    ev.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    ev.payload.ullValue = value;
#ifndef DISABLE_NVTX_MARKERS
    nvtxMarkEx(&ev);
#endif
  }

  static void MarkDouble(double value, const char* message = "Marker",
                         int color = 0, int category = 0) {
    nvtxEventAttributes_t ev = CreateEvent(message, color, category);
    ev.payloadType = NVTX_PAYLOAD_TYPE_DOUBLE;
    ev.payload.dValue = value;
#ifndef DISABLE_NVTX_MARKERS
    nvtxMarkEx(&ev);
#endif
  }

  static void MarkWorkitems(uint64_t items, const char* message) {
    MarkUnsignedInt(items, message, 0, CATEGORY_KERNEL_WORKITEMS);
  }
};

struct RangeMarker {
  bool m_running;
  nvtxEventAttributes_t m_ev;
  nvtxRangeId_t m_id;

  explicit RangeMarker(bool autostart = true, const char* message = "Range",
                       int color = 0, int category = 0)
      : m_running(false), m_id(0) {
    m_ev = Marker::CreateEvent(message, color, category);
    if (autostart)
      Start();
  }

  virtual ~RangeMarker() {
    if (m_running)
      Stop();
  }

  void Start() {
#ifndef DISABLE_NVTX_MARKERS
    if (!m_running) {
      m_id = nvtxRangeStartEx(&m_ev);
      m_running = true;
    }
#endif
  }

  void Stop() {
#ifndef DISABLE_NVTX_MARKERS
    if (m_running) {
      nvtxRangeEnd(m_id);
      m_running = false;
      m_id = 0;
    }
#endif
  }
};

struct IntervalRangeMarker : public RangeMarker {
  explicit IntervalRangeMarker(uint64_t workitems,
                               const char* message = "Range",
                               bool autostart = true)
      : RangeMarker(false, message, 0, CATEGORY_INTERVAL_WORKITEMS) {
    this->m_ev.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    this->m_ev.payload.ullValue = workitems;
    if (autostart)
      Start();
  }

  virtual ~IntervalRangeMarker() {}
};
}  // namespace rayjoin

#endif  // UTIL_MARKERS_H
