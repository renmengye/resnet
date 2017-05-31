py_library(
    name = "nnlib",
    srcs = ["nnlib.py"],
    deps = ["//resnet/utils:logger"],
)

py_library(
    name = "multi_tower_model",
    srcs = ["multi_tower_model.py"],
    deps = ["//resnet/utils:logger", ":nnlib"],
)

py_library(
    name = "multi_tower_model_nccl",
    srcs = ["multi_tower_model_nccl.py"],
    deps = ["//resnet/utils:logger", ":nnlib", ":multi_tower_model"],
)

py_library(
    name = "multi_pass_optimizer",
    srcs = ["multi_pass_optimizer.py"],
    deps = ["//resnet/utils:logger"],
)

py_test(
    name = "multi_pass_optimizer_tests",
    srcs = ["multi_pass_optimizer_tests.py"],
    deps = [":multi_pass_optimizer"],
)

py_library(
    name = "multi_pass_model",
    srcs = ["multi_pass_model.py"],
    deps = ["//resnet/utils:logger", ":multi_pass_optimizer"],
)

py_library(
    name = "multi_pass_model_v2",
    srcs = ["multi_pass_model_v2.py"],
    deps = [":nnlib", ":multi_pass_model"],
)

py_library(
    name = "model_factory",
    srcs = ["model_factory.py"],
    deps = [":multi_tower_model", ":multi_tower_model_nccl", 
            ":multi_pass_model", 
            ":multi_pass_model_v2"],
)

py_library(
    name = "resnet_model",
    srcs = ["resnet_model.py"],
    deps = [":nnlib", ":model_factory", "//resnet/utils:logger"],
)

py_test(
    name = "resnet_model_tests",
    srcs = ["resnet_model_tests.py"],
    deps = [
        "//resnet/utils:logger",
        "//resnet/utils:test_utils",
        "//resnet/configs:test_configs",
        "//resnet/configs",
        ":model_factory",
        ":resnet_model",
    ]
)

py_library(
    name = "resnet_model_gpu",
    srcs = ["resnet_model_gpu.py"],
    deps = [":nnlib", ":model_factory", ":resnet_model"],
)


py_library(
    name = "models",
    srcs = ["__init__.py"],
    deps = [":resnet_model", ":resnet_model_gpu", ":model_factory"],
    visibility = ["//visibility:public"],
)
