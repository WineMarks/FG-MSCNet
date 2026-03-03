import lmdb
import cv2
import numpy as np

# 修改为你的实际路径，例如 'DocTamperV1-TrainingSet'
lmdb_path = 'DocTamperV1-TrainingSet'

try:
    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        print(f"✅ 成功打开 LMDB: {lmdb_path}")
        print("🔍 正在读取前 5 个 Key-Value 对...")

        count = 0
        for key, value in cursor:
            key_str = key.decode('utf-8')
            # 打印 Key
            print(f"Key: {key_str}, Value size: {len(value)} bytes")

            # 尝试解码第一张图片看看是否正常
            if count == 1 and 'image' in key_str:
                img_buf = np.frombuffer(value, dtype=np.uint8)
                img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"   📸 图片解码成功! 尺寸: {img.shape}")
                else:
                    print("   ❌ 图片解码失败 (可能不是图片数据)")

            count += 1
            if count >= 10: break

        print(f"📊 总数据量预估: {txn.stat()['entries']}")
except Exception as e:
    print(f"❌ 读取错误: {e}")
    print("请确认是否安装了 lmdb: pip install lmdb")