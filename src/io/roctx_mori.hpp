// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License (see repository LICENSE).
// ============================================================================
// ADDITIVE, env-gated roctx markers for the MORI-IO HOST RDMA send path.
//
// TWO independent, additive instrumentations (each its own env gate; default OFF):
//
//   (1) MORI_ROCTX=1  -> SYNCHRONOUS push/pop ranges around the host ibv_post_send
//       loop (IOEngine[Session]::BatchWrite + RdmaBatchReadWrite). These measure
//       only the HOST POST cost (building WRs + ringing the NIC doorbell). They
//       are stack/same-thread ranges (MoriRoctxRange RAII) and CANNOT span the
//       async post->completion window. Marker names: mori.io.engine_batch_write,
//       mori.rdma.batch_post.{write,read}.
//
//   (2) MORI_ROCTX_TRANSFER=1  -> ASYNCHRONOUS post->CQ ranges that measure the
//       REAL KV transfer/wire duration: started when a *signaled* WR is posted
//       (RdmaBatchReadWrite, needSignal branch) and stopped when its completion
//       is reaped on the CQ (NotifManager::ProcessOneCqe -> ledger->ReleaseByCqe).
//       Uses the PROCESS-WIDE async roctx API roctxRangeStartA/roctxRangeStop
//       (start on the posting thread, stop on the CQ-poll thread). Marker name:
//       mori.rdma.kv_transfer (its own dedicated trace lane).
//
//       RDMA uses SELECTIVE SIGNALING: only the tail WR of each post batch sets
//       IBV_SEND_SIGNALED and receives a SubmissionLedger recordId (== wr_id).
//       Only that signaled WR produces a CQE, so we start exactly ONE async range
//       per signaled WR (keyed by the ledger recordId) -> every started range has
//       a matching stop (the CQE, or the not-posted cleanup path). recordId is
//       per-EP-ledger (not globally unique), so the range map is keyed by the
//       PAIR (SubmissionLedger*, recordId), which is globally unique and identical
//       at the post site (eps[i].ledger) and the CQ site (ep.ledger) because both
//       hold the same shared SubmissionLedger instance.
//
//       In addition, collision-safe keyed instant marks are emitted immediately
//       before ibv_post_send and when the signaled tail CQE is reaped:
//       mori.rdma.qp_post / mori.rdma.qp_cqe. A process-wide monotonic key joins
//       the endpoints without relying on cross-thread range correlation.
//
// CRITICAL: rocprofv3 (rocprofiler-sdk) --marker-trace only intercepts the
// rocprofiler-sdk ROCTx library librocprofiler-sdk-roctx.so, NOT legacy
// libroctx64.so. We dlopen the sdk lib at runtime (RTLD_GLOBAL) and resolve the
// roctx symbols from it (no link-time dependency added to libmori_io.so).
//
// Fully gated + exception-safe: when neither gate is set the lib is never dlopen'd
// and every call is a no-op (a single bool check).
// ============================================================================
#pragma once

#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace mori {
namespace io {
namespace roctx_detail {

using roctx_range_push_t = int (*)(const char*);
using roctx_range_pop_t = int (*)();
using roctx_mark_t = void (*)(const char*);
// Process-wide async range API (start on one thread, stop from any other).
using roctx_range_id_t = std::uint64_t;
using roctx_range_start_t = roctx_range_id_t (*)(const char*);
using roctx_range_stop_t = void (*)(roctx_range_id_t);

inline bool GateOn(const char* name) {
  const char* g = std::getenv(name);
  if (g == nullptr) return false;
  const char c = g[0];
  return (c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y' || c == 'o' || c == 'O');
}

struct RoctxApi {
  bool enabled = false;           // MORI_ROCTX: push/pop host-post anchors
  bool transfer_enabled = false;  // MORI_ROCTX_TRANSFER: async post->cq ranges
  bool transfer_marks_enabled = false;
  roctx_range_push_t push = nullptr;
  roctx_range_pop_t pop = nullptr;
  roctx_mark_t mark = nullptr;
  roctx_range_start_t range_start = nullptr;
  roctx_range_stop_t range_stop = nullptr;

  RoctxApi() {
    const bool want_post = GateOn("MORI_ROCTX");
    const bool want_transfer = GateOn("MORI_ROCTX_TRANSFER");
    if (!want_post && !want_transfer) return;
    // sdk-roctx ONLY (the lib rocprofv3 --marker-trace intercepts).
    void* h = dlopen("librocprofiler-sdk-roctx.so", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) h = dlopen("librocprofiler-sdk-roctx.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) return;
    push = reinterpret_cast<roctx_range_push_t>(dlsym(h, "roctxRangePushA"));
    pop = reinterpret_cast<roctx_range_pop_t>(dlsym(h, "roctxRangePop"));
    mark = reinterpret_cast<roctx_mark_t>(dlsym(h, "roctxMarkA"));
    range_start = reinterpret_cast<roctx_range_start_t>(dlsym(h, "roctxRangeStartA"));
    range_stop = reinterpret_cast<roctx_range_stop_t>(dlsym(h, "roctxRangeStop"));
    enabled = want_post && (push != nullptr && pop != nullptr);
    transfer_enabled = want_transfer && (range_start != nullptr && range_stop != nullptr);
    transfer_marks_enabled = want_transfer && (mark != nullptr);
  }
};

inline RoctxApi& api() {
  static RoctxApi a;  // gate read + dlopen happen exactly once per process
  return a;
}

// (SubmissionLedger*, recordId) -> async roctx range id. recordId is unique only
// within one ledger, so the ledger pointer disambiguates across endpoints.
using TransferKey = std::pair<std::uintptr_t, std::uint64_t>;
struct TransferKeyHash {
  std::size_t operator()(const TransferKey& k) const {
    std::size_t h1 = std::hash<std::uintptr_t>{}(k.first);
    std::size_t h2 = std::hash<std::uint64_t>{}(k.second);
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};
struct TransferRecord {
  roctx_range_id_t range_id{0};
  std::uint64_t submission_key{0};
  std::uint64_t transfer_id{0};
  std::uint64_t bytes{0};
  std::uint64_t wrs{0};
  std::uint64_t merged{0};
  std::uint64_t qp{0};
  bool is_read{false};
};
struct TransferRanges {
  std::mutex mu;
  std::unordered_map<TransferKey, TransferRecord, TransferKeyHash> ranges;
};
inline TransferRanges& transfer_ranges() {
  static TransferRanges t;
  return t;
}

inline std::uint64_t NextSubmissionKey() {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

inline std::string QpEventName(const char* event, const TransferRecord& rec) {
  return std::string(event) + " key=" + std::to_string(rec.submission_key) +
         " id=" + std::to_string(rec.transfer_id) + " qp=" + std::to_string(rec.qp) +
         " bytes=" + std::to_string(rec.bytes) + " wrs=" + std::to_string(rec.wrs) +
         " merged=" + std::to_string(rec.merged) + (rec.is_read ? " op=read" : " op=write");
}

}  // namespace roctx_detail

// RAII range: pushes on construction, pops on destruction (handles every return
// path + exception). No-op when MORI_ROCTX is off. (HOST-POST anchor only.)
class MoriRoctxRange {
 public:
  explicit MoriRoctxRange(const char* name) {
    auto& a = roctx_detail::api();
    if (a.enabled) {
      a.push(name);
      active_ = true;
    }
  }
  MoriRoctxRange(const char* name, uint64_t id) {
    auto& a = roctx_detail::api();
    if (a.enabled) {
      std::string s = std::string(name) + " id=" + std::to_string(id);
      a.push(s.c_str());
      active_ = true;
    }
  }
  // ADDITIVE: host-post anchor variant carrying the whole-call payload size.
  // Keeps id= LAST so end-anchored id= parsers stay valid: "<name> bytes=<N> id=<id>".
  MoriRoctxRange(const char* name, uint64_t id, uint64_t bytes) {
    auto& a = roctx_detail::api();
    if (a.enabled) {
      std::string s = std::string(name) + " bytes=" + std::to_string(bytes) +
                      " id=" + std::to_string(id);
      a.push(s.c_str());
      active_ = true;
    }
  }
  // ADDITIVE: host-post anchor variant that also carries the whole-call WR
  // count (pre-merge request count, i.e. sizes.size() at the RdmaBatchReadWrite
  // call site -- the same "known at entry, from the sizes vector" granularity
  // already used for bytes above). Keeps id= LAST:
  // "<name> bytes=<N> wrs=<M> id=<id>".
  MoriRoctxRange(const char* name, uint64_t id, uint64_t bytes, uint64_t wrs) {
    auto& a = roctx_detail::api();
    if (a.enabled) {
      std::string s = std::string(name) + " bytes=" + std::to_string(bytes) +
                      " wrs=" + std::to_string(wrs) + " id=" + std::to_string(id);
      a.push(s.c_str());
      active_ = true;
    }
  }
  ~MoriRoctxRange() {
    if (active_) {
      auto& a = roctx_detail::api();
      if (a.pop != nullptr) a.pop();
    }
  }
  MoriRoctxRange(const MoriRoctxRange&) = delete;
  MoriRoctxRange& operator=(const MoriRoctxRange&) = delete;

 private:
  bool active_ = false;
};

inline void MoriRoctxMark(const std::string& msg) {
  auto& a = roctx_detail::api();
  if (a.enabled && a.mark != nullptr) a.mark(msg.c_str());
}

// --- ASYNC post->cq KV-transfer ranges (MORI_ROCTX_TRANSFER) ------------------
// Start an async range for a SIGNALED WR at post time. Keyed by (ledger,recordId).
// ADDITIVE: `wrs` carries epWrsSinceSignal[epId] at the signal point -- the
// number of WRs (across possibly several RdmaBatchReadWrite calls, since
// unsignaled chunks roll forward until the next signal) this one signaled
// completion covers. Placed BEFORE id= (same rule as bytes=) so end-anchored
// id= parsers keep matching: "<name> bytes=<N> wrs=<M> id=<id>".
// ADDITIVE: `merged` carries epMergedSinceSignal[epId] at the signal point --
// the pre-merge logical request count coalesced into this signaled segment's
// WRs (same accumulation window as wrs=). Placed BEFORE id= (same rule as
// bytes=/wrs=): "<name> bytes=<N> wrs=<M> merged=<K> id=<id>".
// ADDITIVE: `qp` carries the per-endpoint index (epIdx) of the QP that posted
// this signaled WR. Placed BEFORE id= (same rule as bytes=/wrs=/merged=):
// "<name> bytes=<N> wrs=<M> merged=<K> qp=<Q> id=<id>".
inline void MoriRoctxTransferStart(const void* ledger, std::uint64_t recordId,
                                   std::uint64_t transferId, bool isRead,
                                   std::uint64_t bytes = 0, std::uint64_t wrs = 0,
                                   std::uint64_t merged = 0, std::uint64_t qp = 0) {
  auto& a = roctx_detail::api();
  if ((!a.transfer_enabled && !a.transfer_marks_enabled) || ledger == nullptr) return;
  // bytes=/wrs=/merged=/qp= placed BEFORE id= so the end-anchored id= parsers keep matching.
  std::string s =
      std::string(isRead ? "mori.rdma.kv_transfer.read" : "mori.rdma.kv_transfer") +
      " bytes=" + std::to_string(bytes) + " wrs=" + std::to_string(wrs) +
      " merged=" + std::to_string(merged) + " qp=" + std::to_string(qp) +
      " id=" + std::to_string(transferId);
  roctx_detail::TransferRecord rec;
  rec.range_id = a.transfer_enabled ? a.range_start(s.c_str()) : 0;
  rec.submission_key = roctx_detail::NextSubmissionKey();
  rec.transfer_id = transferId;
  rec.bytes = bytes;
  rec.wrs = wrs;
  rec.merged = merged;
  rec.qp = qp;
  rec.is_read = isRead;
  auto& t = roctx_detail::transfer_ranges();
  {
    std::lock_guard<std::mutex> lk(t.mu);
    t.ranges[{reinterpret_cast<std::uintptr_t>(ledger), recordId}] = rec;
  }
  // This is deliberately the last instrumentation call before ibv_post_send.
  // Unlike the retained async range, the instant mark has no cross-thread TLS state.
  if (a.transfer_marks_enabled) {
    a.mark(roctx_detail::QpEventName("mori.rdma.qp_post", rec).c_str());
  }
}

// Stop the async range for a completed/cleaned-up signaled WR. Idempotent: a
// no-op if no range was started for this (ledger,recordId) (e.g. unsignaled WRs,
// notification CQEs). The roctxRangeStop call is made OUTSIDE the map lock.
inline void MoriRoctxTransferStop(const void* ledger, std::uint64_t recordId,
                                  bool emitCompletionMark = true) {
  auto& a = roctx_detail::api();
  if ((!a.transfer_enabled && !a.transfer_marks_enabled) || ledger == nullptr) return;
  roctx_detail::TransferRecord rec;
  bool found = false;
  {
    auto& t = roctx_detail::transfer_ranges();
    std::lock_guard<std::mutex> lk(t.mu);
    auto it = t.ranges.find({reinterpret_cast<std::uintptr_t>(ledger), recordId});
    if (it != t.ranges.end()) {
      rec = it->second;
      t.ranges.erase(it);
      found = true;
    }
  }
  if (!found) return;
  if (a.transfer_enabled && a.range_stop != nullptr) a.range_stop(rec.range_id);
  if (emitCompletionMark && a.transfer_marks_enabled) {
    a.mark(roctx_detail::QpEventName("mori.rdma.qp_cqe", rec).c_str());
  }
}

// Diagnostics: number of started-but-not-stopped transfer ranges (leak counter).
inline std::size_t MoriRoctxTransferOutstanding() {
  auto& a = roctx_detail::api();
  if (!a.transfer_enabled && !a.transfer_marks_enabled) return 0;
  auto& t = roctx_detail::transfer_ranges();
  std::lock_guard<std::mutex> lk(t.mu);
  return t.ranges.size();
}

}  // namespace io
}  // namespace mori
