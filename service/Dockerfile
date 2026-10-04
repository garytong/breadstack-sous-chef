# sous-chef service image. Build context = service/ (the Go module):
#
#   docker build -f service/Dockerfile --build-arg BUILD_ID=$(git rev-parse --short HEAD) -t sous-chef:dev service
#
# One static binary, no runtime files: the migrations are embedded
# (migrations/embed.go, //go:embed *.sql), so `-migrate` works from the same
# image with nothing copied beside the binary. The Cloud Run jobs
# (infra/modules/sous-chef/job.tf) set `args` only, so ENTRYPOINT is the
# binary in exec form and `-migrate` / `-retention` arrive as its arguments.

# --- build -------------------------------------------------------------------
FROM golang:1.24 AS build
WORKDIR /src
# go.mod says `go 1.24` with no toolchain line; never download another one.
ENV GOTOOLCHAIN=local CGO_ENABLED=0 GOOS=linux
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# BUILD_ID is what /version reports (main.buildID; the BUILD_ID env var at
# run time wins over it). CI passes the git tag or short sha.
ARG BUILD_ID=dev
RUN go build -trimpath -ldflags "-s -w -X main.buildID=${BUILD_ID}" -o /out/sous-chef ./cmd/sous-chef

# --- runtime -----------------------------------------------------------------
# distroless/static: CA certificates (Anthropic API, thermometer, Cloud SQL
# connector TLS), tzdata and /etc/passwd with the `nonroot` user (uid 65532);
# no shell, no package manager. The binary is static (CGO_ENABLED=0).
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/sous-chef /sous-chef
USER nonroot:nonroot
EXPOSE 8090
ENTRYPOINT ["/sous-chef"]
