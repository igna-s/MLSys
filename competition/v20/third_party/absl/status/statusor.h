// Local stub for third_party/absl/status/statusor.h
//
// The contest infrastructure provides the real Abseil headers when it builds
// and links the evaluator. This stub exists ONLY so the official mlsys.h
// (which forward-declares ReadProblem / ReadSolution / Evaluate returning
// absl::StatusOr<...>) parses during local offline compilation of the solver.
//
// The solver never calls these three functions, so no real implementation or
// linkage against Abseil is required for this build.
#ifndef THIRD_PARTY_ABSL_STATUS_STATUSOR_H_
#define THIRD_PARTY_ABSL_STATUS_STATUSOR_H_

#include <utility>

namespace absl {

template <typename T>
class StatusOr {
 public:
  StatusOr();
  StatusOr(const T&);
  StatusOr(T&&);

  bool ok() const;
  const T& value() const&;
  T& value() &;
  T&& value() &&;

  const T& operator*() const&;
  T& operator*() &;

  const T* operator->() const;
  T* operator->();
};

}  // namespace absl

#endif  // THIRD_PARTY_ABSL_STATUS_STATUSOR_H_
