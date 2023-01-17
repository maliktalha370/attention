def get_config():
    parser = argparse.ArgumentParser()
    # Dataset args
    parser.add_argument("--input_size", type=int, default=224, help="input size")
    parser.add_argument("--output_size", type=int, default=64, help="output size")
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")

    parser.add_argument("--head_da", default=False, action="store_true", help="Do DA on head backbone")
    parser.add_argument("--rgb_depth_da", default=False, action="store_true", help="Do DA on rgb/depth backbone")
    parser.add_argument("--channels_last", default=False, action="store_true")
    parser.add_argument("--freeze_scene", default=False, action="store_true", help="Freeze the scene backbone")
    parser.add_argument("--freeze_face", default=False, action="store_true", help="Freeze the head backbone")
    parser.add_argument("--freeze_depth", default=False, action="store_true", help="Freeze the depth backbone")
    args = parser.parse_args()

    # Print configuration
    print(vars(args))
    return args
