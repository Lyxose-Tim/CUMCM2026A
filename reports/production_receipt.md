# 生产回执（production receipt）

生成时间：2026-09-12 10:23:24；代码提交 4ab08bed47c6ee7addf0a9caa894de8c9b2ce0c5
config_id：D12-20260912-N1600.800.800-integral8pt-bdf1e-8；approved=True

## 实际生效分问配置

| 问 | final_N | 界面/求积 | rtol/atol_C/atol_T |
|---|---|---|---|
| q1 | 1600 | integral/8点 | 1e-08/1e-11/1e-08 |
| q23 | 800 | integral/8点 | 1e-08/1e-11/1e-08 |
| q4 | 800 | integral/8点 | 1e-08/1e-11/1e-08 |

## 达标与采样（未舍入）

- Q2/Q3：t* = 206906.496833 s（57.474027 h）；t_end_1s = 206907 s；t_sample = 206940 s；续算 post_max_cmax = 0.150000000
- Q4：t* = 183931.302883 s（51.092029 h）；t_sample = 183960 s；续算 post_max_cmax = 0.150000000

## 官方文件与哈希（SHA-256）

- `outputs/result1.xlsx`：`b7270ed1b5eec10ace4a0ef2d70ef7294e4ef92e792353c232f010502f7ccdee`
- `outputs/result2.xlsx`：`387542f8c9ca09d0dd1af6e3547d47b54d90b0f69b233d4e5943cc13e17de675`
- `outputs/result3.xlsx`：`6371fbe98dc5f8e3669c15fe537692e36ea9fe09719af8da81d9fab3c1797ecd`
- `outputs/result4.xlsx`：`fb55c152c5c86d693e0342162f78abb25b0011df98fccec5d0b2093ac119f361`

## V-8/V-9 终检

- V-9 result1：通过
- V-9 result2：通过
- V-9 result3：通过
- V-9 result4：通过
- V-8 跨文件：通过（共同时刻 3448，不一致 0，覆盖缺口 0）

总体：✅ 生产验收通过