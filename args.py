import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description="Your program's description here")

    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--accumulative_batch_size', type=int, default=2, help='Accumulative batch size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--slice_per_image', type=int, default=1, help='Slices per image')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--sample_size', type=int, default=4, help='Sample size')
    parser.add_argument('--image_size', type=int, default=1024, help='Image size')
    parser.add_argument('--run_name', type=str, default='debug', help='The name of the run')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--batch_step_one', type=int, default=15, help='Batch one')
    parser.add_argument('--batch_step_two', type=int, default=25, help='Batch two')
    parser.add_argument('--conv_model', type=str, default=None, help='Path to convolution model')
    parser.add_argument('--custom_bias', type=float, default=0, help='Learning Rate')
    parser.add_argument("--inference", action="store_true", help="Set for inference")
    ########################################################################################
    parser.add_argument("--train_dir",type=str, help="Path to the training data")
    parser.add_argument("--test_dir",type=str, help="Path to the test data")
    

    return parser.parse_args()