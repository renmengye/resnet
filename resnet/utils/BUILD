py_library(
    name = "logger",
    srcs = ["logger.py"],
    deps = [],
    visibility = ["//visibility:public"],
)

py_library(
    name = "experiment_logger",
    srcs = ["experiment_logger.py"],
    deps = [":logger"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "lr_schedule",
    srcs = ["lr_schedule.py"],
    deps = [":logger"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "batch_iter",
    srcs = ["batch_iter.py", "concurrent_batch_iter.py"],
    deps = [":logger"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "gen_id",
    srcs = ["gen_id.py"],
    deps = [],
    visibility = ["//visibility:public"],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [":logger"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "utils",
    srcs = ["__init__.py"],
    deps = [":logger", ":experiment_logger", ":lr_schedule", ":batch_iter",
            ":gen_id"],
    visibility = ["//visibility:public"],
)
