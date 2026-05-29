import os
import shutil
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Compress GRPO checkpoint and copy to Google Drive")
    parser.add_argument("--checkpoint_dir", type=str, default="./grpo_cot_output/final_lora", help="Path to the checkpoint folder to compress")
    parser.add_argument("--output_zip", type=str, default="./grpo_lora_checkpoint", help="Base name of the output zip file (without .zip)")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/grpo_checkpoints", help="Google Drive target folder path")
    args = parser.parse_args()

    # 1. Validate checkpoint path
    if not os.path.exists(args.checkpoint_dir):
        print(f"[-] Checkpoint directory {args.checkpoint_dir} does not exist!")
        sys.exit(1)

    # 2. Compress the folder
    print(f"[*] Compressing {args.checkpoint_dir} to {args.output_zip}.zip...")
    try:
        shutil.make_archive(args.output_zip, 'zip', args.checkpoint_dir)
        zip_file_path = f"{args.output_zip}.zip"
        size_mb = os.path.getsize(zip_file_path) / (1024 * 1024)
        print(f"[+] Compression complete! File size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"[-] Compression failed: {e}")
        sys.exit(1)

    # 3. Check and copy to Google Drive
    drive_mount_root = "/content/drive"
    if not os.path.exists(drive_mount_root):
        print("[!] Warning: Google Drive does not appear to be mounted at /content/drive.")
        print("[!] To mount Google Drive, run this in a Colab cell:")
        print("    from google.colab import drive")
        print("    drive.mount('/content/drive')")
        print(f"[+] Zip file remains saved locally at: {os.path.abspath(zip_file_path)}")
        sys.exit(0)

    # Ensure target drive directory exists
    os.makedirs(args.drive_dir, exist_ok=True)
    destination_path = os.path.join(args.drive_dir, os.path.basename(zip_file_path))
    
    print(f"[*] Copying zip file to Google Drive: {destination_path}...")
    try:
        shutil.copy2(zip_file_path, destination_path)
        print("[+] Upload to Google Drive completed successfully!")
    except Exception as e:
        print(f"[-] Upload to Google Drive failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
