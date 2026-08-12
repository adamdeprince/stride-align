\name{stride_backend}
\alias{stride_backend}
\alias{stride_available_backends}
\title{Inspect stride-align SIMD dispatch}
\usage{
stride_backend()

stride_available_backends()
}
\value{
A character vector. \code{stride_backend()} returns the selected backend;
\code{stride_available_backends()} returns compatible packaged backends in
dispatch-priority order.
}
\description{
Reports the native library selected when the package namespace was loaded.
Set the \code{STRIDE_ALIGN_R_BACKEND} environment variable before loading the
package to pin a compatible packaged backend for benchmarking.
}
