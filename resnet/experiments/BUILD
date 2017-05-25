py_binary(
    name = "run_cifar_exp",
    srcs = ["run_cifar_exp.py"],
    deps = ["//resnet/models", "//resnet/data", "//resnet/utils", 
            "//resnet/configs"],
)

py_binary(
    name = "run_imagenet_exp",
    srcs = ["run_imagenet_exp.py"],
    deps = ["//resnet/models", "//resnet/data", "//resnet/utils", 
            "//resnet/configs"],
)

py_binary(
    name = "run_imagenet_eval",
    srcs = ["run_imagenet_eval.py"],
    deps = ["//resnet/models", "//resnet/data", "//resnet/utils", 
            "//resnet/configs"],
)

py_library(
    name = "experiments",
    srcs = [],
    deps = [":run_cifar_exp", ":run_imagenet_exp", ":run_imagenet_eval"],
)
