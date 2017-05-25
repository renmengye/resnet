py_library(
    name = "config_factory",
    srcs = ["config_factory.py"],
    deps = [],
)

py_library(
    name = "cifar_configs",
    srcs = ["cifar_configs.py"],
    deps = [":config_factory"],
)

py_library(
    name = "test_configs",
    srcs = ["test_configs.py"],
    deps = [":cifar_configs"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "imagenet_configs",
    srcs = ["imagenet_configs.py"],
    deps = [],
)

py_library(
    name = "configs",
    srcs = ["__init__.py"],
    deps = [":config_factory", ":cifar_configs", ":imagenet_configs"],
    visibility = ["//visibility:public"],
)
