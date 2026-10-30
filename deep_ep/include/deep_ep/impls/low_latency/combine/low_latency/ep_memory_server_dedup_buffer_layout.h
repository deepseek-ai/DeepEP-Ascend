// Copyright (c) 2026, Lu Lu
// Modified by candy_cloud_fly 2026

#pragma once
// Existing ascendEP dispatch controls end before 764 KiB. Keep its 2 MiB
// data boundary and fixed 4 GiB bank stride; do not import upstream offsets.
#define COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_CONTROL_OFFSET (764ULL * 1024ULL)
#define COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_PREPARE_DONE_OFFSET 0ULL
#define COMBINE_DEDUP_EP_SERVER_DEDUP_COMBINE_SERVER_READY_OFFSET 16384ULL
#define COMBINE_DEDUP_EP_SERVER_DEDUP_SHARED_DATA_OFFSET (2ULL * 1024ULL * 1024ULL)
