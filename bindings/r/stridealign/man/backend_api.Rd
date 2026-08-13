\name{backend_api}
\alias{BackendKind}
\alias{BackendRecord}
\alias{available_backends}
\alias{backend_is_available}
\alias{detect_best_backend}
\title{Canonical backend inspection API}
\usage{
BackendRecord(kind, compiled = TRUE, available = TRUE)
available_backends()
backend_is_available(kind)
detect_best_backend()
}
\arguments{
\item{kind}{A backend name, usually a value from \code{BackendKind}.}
\item{compiled}{Whether the backend was compiled into the package.}
\item{available}{Whether it can run on this CPU.}
}
\value{
\code{available_backends()} returns a list of backend records;
\code{backend_is_available()} returns one logical value; and
\code{detect_best_backend()} returns the selected backend name.
}
\description{
R-native equivalents of stride-align's Python backend catalog and detection
helpers.
}
