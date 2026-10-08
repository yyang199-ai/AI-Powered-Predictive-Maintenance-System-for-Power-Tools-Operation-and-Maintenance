"""Readable research dashboard: public bearing evidence and a separate virtual lab."""
from io import BytesIO
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "artifacts" / "bearing" / "data"
BUNDLE = ROOT / "artifacts" / "bearing"
PAGES = ["公开轴承预验证", "虚拟电动工具实验台", "模型与使用说明"]
CLASS_NAMES = ["正常轴承", "内圈故障", "滚动体故障", "外圈故障"]
MODEL_NAMES = {"cnn": "轻量 CNN", "softmax": "CNN + 标准 Transformer", "linear": "CNN + 线性注意力"}
PUBLIC_FEATURE_NAMES = {
    "vibration_rms": ("振动均方根（RMS）", "数据集加速度单位", "振动整体强度"),
    "kurtosis": ("峭度", "无量纲", "冲击是否突出"),
    "crest_factor": ("峰值因子", "无量纲", "峰值相对于整体强度的大小"),
    "peak_to_peak": ("峰峰值", "数据集加速度单位", "最高与最低值之差"),
    "spectral_centroid_hz": ("频谱重心", "Hz", "能量分布的平均频率"),
    "peak_frequency_hz": ("主峰频率", "Hz", "振动频谱最突出的频率"),
    "spectral_entropy": ("频谱熵", "无量纲", "能量分布的分散程度"),
    "band_energy_low": ("低频能量占比（0–1 kHz）", "比例", "低频段占总能量的比例"),
    "band_energy_mid": ("中频能量占比（1–3 kHz）", "比例", "中频段占总能量的比例"),
    "band_energy_high": ("高频能量占比（3–6 kHz）", "比例", "高频段占总能量的比例"),
    "envelope_peak_frequency_hz": ("包络谱主峰频率", "Hz", "振幅变化的周期信息"),
}
LAB_FEATURE_NAMES = {
    "vibration_raw_rms": ("原始振动均方根", "m/s²"), "vibration_rms": ("去噪后振动均方根", "m/s²"),
    "vibration_kurtosis": ("振动峭度", "无量纲"), "vibration_peak_hz": ("振动主峰频率", "Hz"),
    "vibration_spectral_centroid_hz": ("振动频谱重心", "Hz"), "vibration_fft_resolution_hz": ("频率分辨率", "Hz"),
    "vibration_low_band_power": ("振动低频功率（<1 kHz）", "(m/s²)²"),
    "vibration_high_band_power": ("振动高频功率（≥1 kHz）", "(m/s²)²"),
    "vibration_total_spectral_power": ("振动频谱总功率", "(m/s²)²"),
    "current_mean": ("平均电流", "A"), "current_rms": ("电流均方根", "A"), "current_ripple": ("电流纹波标准差", "A"),
    "temperature_rise": ("本次记录温升", "℃"), "temperature_mean_c": ("平均温度", "℃"),
    "temperature_slope": ("温度变化斜率", "℃/秒"), "shaft_frequency_hz": ("设定轴转频", "Hz"),
    "current_harmonic_1_amplitude": ("电流一阶转频幅值", "A"), "current_harmonic_2_amplitude": ("电流二阶转频幅值", "A"),
    "current_harmonic_3_amplitude": ("电流三阶转频幅值", "A"), "current_harmonic_distortion_ratio": ("高阶幅值与一阶的比值", "无量纲"),
}

torch.set_num_threads(2)
st.set_page_config(page_title="电动工具实验与诊断平台", layout="wide")


@st.cache_data(show_spinner=False)
def read_dataset(path, modified, manifest_modified):
    del modified, manifest_modified
    from predictive.public_data import sha256_file
    manifest = read_json(Path(path).parent / "dataset_manifest.json")
    if not manifest.get("dataset_sha256") or sha256_file(path) != manifest["dataset_sha256"]:
        raise ValueError("数据包 SHA-256 校验不一致，请重新准备数据。")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


@st.cache_data(show_spinner=False)
def read_table(path, modified):
    del modified
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_verified_features(path, modified, expected_hash):
    del modified
    from predictive.public_data import sha256_file
    if sha256_file(path) != expected_hash:
        raise ValueError("特征表 SHA-256 校验不一致。")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_nasa(path, modified):
    del modified
    from predictive.ims_data import load_ims_prepared
    return load_ims_prepared(path)


@st.cache_resource(show_spinner=False)
def read_model(path, modified):
    del modified
    from predictive.bearing_model import load_bearing_bundle
    return load_bearing_bundle(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_manifest_path():
    return BUNDLE / "manifest.json"


def setup_commands():
    st.code("python scripts/prepare_public_data.py\npython scripts/train_bearing.py --epochs 15\npython -m streamlit run app.py", language="bash")
    st.caption("在项目文件夹中、已安装项目依赖的 Python 环境执行；Windows 可使用项目提供的启动脚本。")


def draw_signal(values, rate, label, unit, max_points=2000, time_label="窗口内相对时间（秒）"):
    values = np.asarray(values).reshape(-1)
    indexes = np.unique(np.linspace(0, len(values) - 1, min(len(values), max_points)).astype(int))
    display = pd.DataFrame({f"{label}（{unit}）": values[indexes]}, index=indexes / rate)
    display.index.name = time_label
    st.line_chart(display, use_container_width=True)


def draw_spectrum(values, rate, unit, limit_hz=None):
    values = np.asarray(values).reshape(-1)
    frequencies = np.fft.rfftfreq(len(values), d=1 / rate)
    amplitudes = np.abs(np.fft.rfft(values - values.mean())) / len(values)
    if len(amplitudes) > 1:
        amplitudes[1:-1 if len(values) % 2 == 0 else None] *= 2
    keep = np.ones(len(frequencies), dtype=bool)
    if limit_hz is not None:
        keep = frequencies <= limit_hz
    display = pd.DataFrame({f"单边幅值（{unit}）": amplitudes[keep]}, index=frequencies[keep])
    display.index.name = "频率（Hz）"
    st.line_chart(display, use_container_width=True)


def reset_bearing():
    st.session_state.bearing_window_number = 1
    st.session_state.bearing_pending = False
    st.session_state.bearing_result = None
    st.session_state.bearing_result_identity = None
    st.session_state.bearing_status = "尚未开始"


def request_analysis():
    st.session_state.bearing_pending = True


def next_bearing():
    current = st.session_state.get("bearing_window_number", 1)
    maximum = st.session_state.get("bearing_window_count", 1)
    st.session_state.bearing_window_number = min(current + 1, maximum)
    st.session_state.bearing_pending = True


def public_page():
    st.header("公开轴承数据：先验证能否识别故障部位")
    st.info("这是凯斯西储大学（CWRU）轴承试验台的历史振动数据。模型识别正常、内圈、滚动体和外圈故障；本页没有连接电动工具。")
    model_path = model_manifest_path()
    if model_path.exists():
        try:
            summary = read_json(model_path)
            selected_test = summary.get("experiments", {}).get(summary.get("selected_model"), {}).get("test", {})
            normal_false_positive = selected_test.get("healthy_window_false_positive_rate")
            if normal_false_positive is not None and normal_false_positive > 0.1:
                st.warning(f"当前留出的正常轴承记录误判较多：正常窗口误判为故障的比例为 {normal_false_positive:.1%}。本页用于公开台架算法预验证，当前不能直接用于真实运维告警。")
        except (ValueError, OSError):
            pass
    with st.expander("第一次使用：按这个顺序操作", expanded=True):
        st.markdown("1. 选择一条**测试集记录**和一个窗口。\n2. 点击 **开始分析当前窗口**，查看模型判断与数据集真实标签是否一致。\n3. 点击 **下一窗口并分析** 做手动历史回放；下方波形和频谱帮助理解输入信号。")
        st.caption("缺陷尺寸档来自数据集的人工加工尺寸，不是模型预测出的自然磨损等级。当前模型不能给出真实剩余寿命。")

    dataset_path = DATA / "dataset.npz"
    if not dataset_path.exists():
        st.error("公开数据尚未准备完成，当前无法进行公开数据预验证。")
        setup_commands()
        return
    try:
        data_manifest_path = DATA / "dataset_manifest.json"
        dataset = read_dataset(str(dataset_path), dataset_path.stat().st_mtime_ns, data_manifest_path.stat().st_mtime_ns)
        required = {"X", "y", "split", "record_id", "start_seconds", "defect_diameter_mm"}
        if required - dataset.keys():
            raise ValueError("缺少字段：" + ", ".join(sorted(required - dataset.keys())))
    except (ValueError, OSError, KeyError) as error:
        st.error("公开数据文件无法读取：" + str(error))
        return
    st.caption("已核对预处理数据包的 SHA-256 校验值；原始数据来源与官网复核情况见“下载与追溯”。")

    left, right = st.columns([2, 1])
    with left:
        test_mask = np.asarray(dataset["split"]).astype(str) == "test"
        records = sorted(set(np.asarray(dataset["record_id"])[test_mask].astype(str)))
        if not records:
            st.error("数据集没有测试集记录，请重新准备数据。")
            return
        record = st.selectbox("测试集记录（从未参与训练）", records, key="bearing_record")
    with right:
        st.caption("数据来源：公开轴承试验台\n\n实际可用通道：驱动端振动（DE）\n\n电流、温度：该诊断输入未提供")

    indexes = np.flatnonzero(test_mask & (np.asarray(dataset["record_id"]).astype(str) == record))
    if st.session_state.get("bearing_last_record") != record:
        reset_bearing()
        st.session_state.bearing_last_record = record
    st.session_state.bearing_window_count = len(indexes)
    window_number = st.number_input("记录内窗口（从 1 开始）", min_value=1, max_value=len(indexes), step=1, key="bearing_window_number")
    controls = st.columns(3)
    controls[0].button("开始分析当前窗口", on_click=request_analysis, type="primary", use_container_width=True)
    controls[1].button("下一窗口并分析", on_click=next_bearing, disabled=window_number >= len(indexes), use_container_width=True)
    controls[2].button("重置", on_click=reset_bearing, use_container_width=True)
    index = int(indexes[int(window_number) - 1])
    identity = (record, index)
    waveform = np.asarray(dataset["X"][index]).reshape(-1)
    label = int(dataset["y"][index])
    diameter = float(dataset["defect_diameter_mm"][index])
    damage = "正常（无人工缺陷）" if diameter == 0 else (
        "小尺寸人工缺陷" if diameter <= 0.18 else "中尺寸人工缺陷" if diameter <= 0.36 else "大尺寸人工缺陷"
    )
    if st.session_state.get("bearing_result_identity") != identity:
        st.session_state.bearing_result = None
        if not st.session_state.get("bearing_pending", False):
            st.session_state.bearing_status = "尚未分析此窗口"
    if st.session_state.get("bearing_pending", False):
        st.session_state.bearing_pending = False
        if not np.isfinite(waveform).all():
            st.session_state.bearing_status = "输入质量异常"
            st.error("该窗口含无效数值，已停止判断。")
        elif not model_path.exists():
            st.session_state.bearing_status = "模型尚未就绪"
            st.error("公开数据模型尚未训练完成。请运行下方训练命令。")
        else:
            try:
                from predictive.bearing_model import predict_bearing
                with st.spinner("分析所选公开数据窗口…"):
                    model, manifest = read_model(str(BUNDLE), model_path.stat().st_mtime_ns)
                    prediction = predict_bearing(waveform, model, manifest, prepared=True)[0]
                st.session_state.bearing_result = prediction
                st.session_state.bearing_result_identity = identity
                st.session_state.bearing_status = "本窗口分析完成"
            except (ImportError, ValueError, RuntimeError, OSError, KeyError) as error:
                st.session_state.bearing_status = "分析失败"
                st.error("模型无法分析该窗口：" + str(error))
    prediction = st.session_state.get("bearing_result")
    status = st.session_state.get("bearing_status", "尚未开始")
    short_status = {"本窗口分析完成": "已完成", "尚未开始": "待分析", "尚未分析此窗口": "待分析", "模型尚未就绪": "模型未就绪", "输入质量异常": "输入异常", "分析失败": "失败"}.get(status, status)
    metrics = st.columns(4)
    metrics[0].metric("测试状态", short_status)
    metrics[1].metric("模型判断", prediction.get("class_name_zh", CLASS_NAMES[int(prediction["class_index"])]) if prediction else "等待分析")
    metrics[2].metric("数据集真实标签", CLASS_NAMES[label])
    metrics[3].metric("已知缺陷尺寸", "0 mm" if diameter == 0 else f"{diameter:.4f} mm")
    st.write(f"**受损记录的已知尺寸档：{damage}。** 此字段用于对照，不是模型推断的磨损程度。")
    start_seconds = float(dataset["start_seconds"][index])
    details = [f"记录 {record}", f"窗口 {int(window_number)}/{len(indexes)}", f"起点 {start_seconds:.3f} 秒", "采样率 12,000 Hz", f"输入长度 {len(waveform)} 点"]
    if "rpm" in dataset:
        details.append(f"试验台转速 {float(dataset['rpm'][index]):.0f} r/min")
    if "load_hp" in dataset:
        details.append(f"试验台负载 {int(dataset['load_hp'][index])} HP")
    st.caption(" · ".join(details))
    st.caption("状态说明：" + status + "；当前为手动历史回放，无实时设备连接。")

    if prediction:
        if int(prediction["class_index"]) == label:
            st.success("该窗口的故障部位判断与数据集标签一致。单个窗口结果不能证明真实电动工具的诊断效果。")
        else:
            st.warning("该窗口的模型判断与真实标签不一致，建议将此记录列入误判分析。")
        scores = prediction.get("scores", {})
        if scores:
            names = {"normal": CLASS_NAMES[0], "inner_race": CLASS_NAMES[1], "ball": CLASS_NAMES[2], "outer_race": CLASS_NAMES[3]}
            st.dataframe(pd.DataFrame({"候选类别": [names.get(key, key) for key in scores], "模型分数": [f"{float(value):.1%}" for value in scores.values()]}), hide_index=True, use_container_width=True)
            st.caption("模型分数尚未进行概率校准，不能解释为真实现场的故障概率。")
    elif not model_path.exists():
        st.warning("公开诊断模型尚未就绪；现在可以查看已准备的数据和信号。")
        setup_commands()
    st.caption("剩余寿命：本数据集没有可用于验证自然退化寿命的标签，因此本页不输出寿命倒计时。")

    waveform_tab, feature_tab, download_tab = st.tabs(["输入波形与频谱", "预处理与特征", "下载与追溯"])
    with waveform_tab:
        a, b = st.columns(2)
        with a:
            st.subheader("去均值、Haar 小波阈值处理后的振动")
            draw_signal(waveform, 12000, "振动", "数据集加速度原单位")
        with b:
            st.subheader("振动 FFT 幅值谱")
            draw_spectrum(waveform, 12000, "数据集加速度原单位")
        st.caption("图中为模型输入窗口。时域图看振动随时间变化，频谱看振动能量集中在哪些频率；二者不与电流、温度混用纵轴。")
    with feature_tab:
        st.write("公开 MAT 文件 → 核对记录与采样率 → 统一到 12 kHz → 按完整记录分配训练/验证/测试 → 切窗 → 去均值 → Haar 小波阈值去噪 → 提取时域、FFT 与包络特征。神经网络输入使用去噪波形。")
        st.caption("训练集均值、标准差只从训练数据拟合；测试窗口没有参与参数学习。记录划分不能保证不同文件来自不同物理轴承，仍需独立设备验证。")
        features_path = DATA / "features.csv"
        if features_path.exists():
            try:
                data_manifest = read_json(data_manifest_path)
                features = read_verified_features(str(features_path), features_path.stat().st_mtime_ns, data_manifest["features_sha256"])
                rows = features[features["record_id"].astype(str) == record]
                if "start_seconds" in rows:
                    rows = rows[np.isclose(rows["start_seconds"], start_seconds)]
                if not rows.empty:
                    row = rows.iloc[0]
                    details = [{"特征": label, "数值": f"{float(row[key]):.5g}", "单位": unit, "怎么看": explanation} for key, (label, unit, explanation) in PUBLIC_FEATURE_NAMES.items() if key in row]
                    st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
                    with st.expander("查看完整标签与窗口身份（原始字段）"):
                        st.dataframe(rows.head(1).T.rename(columns={rows.index[0]: "当前窗口"}).astype(str), use_container_width=True)
            except (ValueError, OSError, KeyError) as error:
                st.error("特征表无法验证：" + str(error))
    with download_tab:
        st.download_button("下载可直接训练的数据集 NPZ", dataset_path.read_bytes(), "cwru_prepared_dataset.npz", "application/octet-stream")
        selected = pd.DataFrame({"relative_seconds": np.arange(len(waveform)) / 12000, "vibration_preprocessed": waveform})
        st.download_button("下载当前预处理窗口 CSV", selected.to_csv(index=False).encode("utf-8-sig"), f"cwru_{record}_window_{int(window_number)}.csv", "text/csv")
        for path, label, mime in [
            (DATA / "features.csv", "下载全部窗口特征与标签 CSV", "text/csv"),
            (DATA / "dataset_manifest.json", "下载数据划分与预处理说明", "application/json"),
            (ROOT / "data" / "public" / "CWRU" / "manifest.json", "下载原始文件来源与校验记录", "application/json"),
        ]:
            if path.exists():
                st.download_button(label, path.read_bytes(), path.name, mime)
        st.caption("训练可直接使用 artifacts/bearing/data/dataset.npz（X、y、split、record_id 等字段）。原始文件来源及 SHA-256 校验值见下载清单。")


def lab_page():
    st.header("虚拟电动工具实验台：观察工况与损伤怎样影响信号")
    st.info("本页用物理现象的简化模型生成振动、电流和温度波形，用于搭建流程与设计实验。所有记录均为虚拟数据；设定的损伤等级是生成条件。")
    try:
        from predictive.lab import create_lab_run, extract_lab_features
    except ImportError as error:
        st.error("实验台模块尚未就绪：" + str(error))
        return
    tools = {"电钻": "drill", "角磨机": "grinder", "冲击扳手": "impact_wrench"}
    damages = {"正常": "normal", "轻度（虚拟）": "mild", "中度（虚拟）": "moderate", "重度（虚拟）": "severe"}
    tool_defaults = {"drill": (3600, 1800), "grinder": (16000, 11000), "impact_wrench": (4500, 2200)}
    with st.expander("怎么使用实验台", expanded=True):
        st.write("选择工具、损伤设定、负载和转速，点击生成。先保持其他条件不变，只改变损伤等级，对比信号；再保持损伤等级不变、改变负载或转速，观察工况引起的变化。下载记录用于自己的分析。")
    tool_name = st.selectbox("工具类型", list(tools), key="lab_tool")
    tool = tools[tool_name]
    with st.form("lab_parameters"):
        a, b, c = st.columns(3)
        damage_name = a.selectbox("损伤设定", list(damages))
        load = b.slider("负载比例", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        maximum_rpm, default_rpm = tool_defaults[tool]
        rpm = c.number_input("转速（r/min）", min_value=60, max_value=maximum_rpm, value=default_rpm, step=50, key=f"lab_rpm_{tool}")
        d, e, f = st.columns(3)
        duration = d.number_input("记录时长（秒）", min_value=0.5, max_value=60.0, value=5.0, step=0.5)
        ambient = e.number_input("环境温度（℃）", min_value=-10.0, max_value=50.0, value=25.0, step=1.0)
        seed = f.number_input("随机种子（相同条件可复现）", min_value=0, max_value=999999, value=42, step=1)
        submitted = st.form_submit_button("生成虚拟实验记录", type="primary")
    if submitted:
        try:
            with st.spinner("生成多通道波形并提取实验特征…"):
                run = create_lab_run(tool=tool, damage=damages[damage_name], load=load, rpm=rpm, duration_seconds=duration, seed=int(seed), ambient_temperature_c=ambient)
                st.session_state.lab_run = run
                st.session_state.lab_features = extract_lab_features(run)
                st.session_state.lab_damage_name = damage_name
        except (ValueError, RuntimeError) as error:
            st.error("实验记录生成失败：" + str(error))
    run = st.session_state.get("lab_run")
    if run is None:
        st.metric("实验状态", "待生成")
        st.write("完成上方设置后点击生成，即可看到三个通道的信号和特征。")
        return
    metadata = run.metadata
    metrics = st.columns(4)
    metrics[0].metric("实验状态", "已生成")
    metrics[1].metric("本次工具", metadata.get("tool_name", metadata.get("tool", "")))
    metrics[2].metric("设定损伤等级", st.session_state.get("lab_damage_name", metadata.get("damage", "")))
    metrics[3].metric("本次转速", f"{float(metadata['rpm']):.0f} r/min")
    st.caption(f"状态说明：虚拟记录已生成；本次负载比例 {float(metadata['load']):.0%}。修改控件后，需要再次点击生成才会产生新记录。")
    st.warning("上方损伤等级由你设定，尚未经过真实传感器或模型验证。虚拟多通道波形不能直接输入只接受 CWRU 振动的诊断模型；旧六维仿真模型的时间尺度也不同。")
    channel_labels = {"vibration": ("振动", "m/s²"), "current": ("电流", "A"), "temperature": ("温度", "℃")}
    tabs = st.tabs(["振动", "电流", "温度", "特征与导出"])
    for tab, channel in zip(tabs[:3], channel_labels):
        with tab:
            values = getattr(run, channel)
            rate = float(run.sampling_rates[channel])
            label, fallback_unit = channel_labels[channel]
            unit = fallback_unit
            st.write(f"{label}采样率：{rate:g} Hz；共 {len(values):,} 点。")
            draw_signal(values, rate, label, unit, time_label="本次实验相对时间（秒）")
            if channel != "temperature":
                st.write("对应频谱（显示至 5 kHz）")
                draw_spectrum(values, rate, unit, limit_hz=5000)
            else:
                st.caption("温度用于观察热积累趋势。几秒钟记录的温升很小，不能替代真实长时热实验。")
    with tabs[3]:
        features = st.session_state.lab_features
        st.write("振动与电流先做小波去噪，频谱计算去均值，再提取时域/FFT、转频阶次电流幅值；温度提取均值与变化趋势。下表用于流程验证和比较，不是故障诊断结论。")
        st.caption("本次记录温升指末值减首值；温度斜率单位为 ℃/秒。电流阶次以模拟轴转频为基准，不代表市电谐波或已校准的 THD。")
        rows = [{"特征": LAB_FEATURE_NAMES.get(key, (key, ""))[0], "数值": f"{float(value):.6g}", "单位": LAB_FEATURE_NAMES.get(key, (key, ""))[1]} for key, value in features.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        memory = BytesIO()
        np.savez_compressed(memory, **{channel: getattr(run, channel) for channel in channel_labels}, **{f"{channel}_time_seconds": run.times[channel] for channel in channel_labels})
        st.download_button("下载本次多通道波形 NPZ", memory.getvalue(), "virtual_tool_waveforms.npz", "application/octet-stream")
        export_metadata = {**metadata, "sampling_rates_hz": run.sampling_rates}
        st.download_button("下载本次配置与来源 JSON", json.dumps(export_metadata, ensure_ascii=False, indent=2).encode("utf-8"), "virtual_tool_metadata.json", "application/json")
        st.download_button("下载本次特征 CSV", pd.DataFrame([features]).to_csv(index=False).encode("utf-8-sig"), "virtual_tool_features.csv", "text/csv")
        st.caption("批量实验及按通道导出 CSV 可使用 scripts/run_virtual_lab.py；完整使用方法见 docs/phase2/网页使用指南.md。")


def nasa_section():
    features_path = ROOT / "data" / "public" / "NASA_IMS" / "prepared" / "features.csv"
    if not features_path.exists():
        return
    st.subheader("NASA IMS：观察完整实验的退化趋势")
    st.info("这是轴承长时间运行实验的观测趋势。到最后一条记录的倒计时是数据集末端代理标签，不是模型预测出的真实剩余寿命。")
    try:
        table, nasa_manifest = read_nasa(str(features_path.parent), features_path.stat().st_mtime_ns)
        bearing = st.selectbox("NASA 轴承通道", sorted(table["bearing_id"].unique()), key="nasa_bearing")
        selected = table[table["bearing_id"] == bearing].sort_values("elapsed_hours")
        column = "raw_vibration_rms" if "raw_vibration_rms" in selected else "vibration_rms"
        display = selected.set_index("elapsed_hours")[[column]].rename(columns={column: "原始振动 RMS" if column == "raw_vibration_rms" else "去噪后振动 RMS"})
        display.index.name = "从首条记录起壁钟时间差（小时）"
        st.line_chart(display, use_container_width=True)
        st.caption("RMS 为每次振动采样记录的均方根；采样记录之间有时间间隔，此图不代表连续波形。壁钟时差不等于已确认的有效运行时长。当前尚未训练并验证可输出真实剩余寿命的模型。")
        st.warning("最后 2 次快照的四个通道都降到低幅值，可能存在停机或工况变化；不能把末端骤降解释为轴承恢复健康或维修成功。")
        with st.expander("查看 NASA 记录与末端代理标签"):
            st.dataframe(selected[[name for name in ["timestamp_iso", "elapsed_hours", "bearing_id", "raw_vibration_rms", "vibration_rms", "endpoint_proxy_rul_hours", "event_observed", "is_censored", "activity_status"] if name in selected]], hide_index=True, use_container_width=True)
            st.download_button("下载 NASA 特征与代理标签 CSV", features_path.read_bytes(), "nasa_ims_features.csv", "text/csv")
    except (KeyError, ValueError, OSError) as error:
        st.warning("NASA 记录仍在准备或格式需检查：" + str(error))


def explanation_page():
    st.header("模型结构、结果与使用说明")
    st.subheader("当前两种实验分别回答什么问题")
    st.dataframe(pd.DataFrame([
        {"实验": "公开轴承预验证", "输入": "CWRU 实测驱动端振动", "输出": "正常 / 内圈 / 滚动体 / 外圈", "适用范围": "公开轴承记录上的故障部位识别"},
        {"实验": "虚拟电动工具实验台", "输入": "你设定的工具、负载、转速与损伤", "输出": "模拟振动、电流、温度及特征", "适用范围": "演示采集处理流程、设计真实实验"},
        {"实验": "保留的旧仿真演示", "输入": "生成的六维特征与虚拟寿命标签", "输出": "仿真健康、RUL、告警和工单", "适用范围": "体验运维闭环，不能作为公开数据验证"},
    ]), hide_index=True, use_container_width=True)
    st.subheader("公开数据模型怎样计算")
    st.write("每次输入一个 1,024 点振动窗口（12 kHz 时约 0.085 秒）。先去均值、小波阈值去噪，再使用训练集均值/标准差归一化。卷积层提取局部冲击与周期信号；可选注意力编码器汇总不同位置的关系；池化与分类层输出四个类别分数。")
    st.write("标准 Transformer 对每对 token 计算关系；线性注意力通过先汇总 key/value，再与 query 计算，减少随 token 数平方增长的部分。轻量 CNN 是更容易部署的备选。最终采用哪一种由验证集结果和 CPU 基准共同决定，测试集用于独立报告。")
    st.caption("原六维仿真模型确实含 1 层、2 个头、32 维 Transformer 和 1 层 LSTM，合计 17,789 参数。参数少不等于已满足 MCU 延迟要求；真实采集、预处理与目标硬件尚需测量。")
    manifest_path = model_manifest_path()
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            kind = str(manifest.get("selected_model", manifest.get("model_config", {}).get("kind", "")))
            st.success("当前公开诊断模型：" + MODEL_NAMES.get(kind, kind or "见模型清单"))
            rows = []
            for name, result in manifest.get("experiments", {}).items():
                test = result.get("test", {})
                profile = result.get("profile", {})
                benchmark = result.get("post_training_benchmark", result.get("benchmark", {}))
                rows.append({"模型": MODEL_NAMES.get(name, name), "测试准确率": f"{float(test['accuracy']):.2%}" if test.get("accuracy") is not None else "未完成", "测试宏平均 F1": f"{float(test['macro_f1']):.3f}" if test.get("macro_f1") is not None else "未完成", "参数量": profile.get("parameter_count"), "CPU 前向 P95（ms）": benchmark.get("forward_p95_ms"), "CPU 处理+推理 P95（ms）": benchmark.get("pipeline_p95_ms")})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("CPU 耗时不是 MCU 实测耗时；处理+推理不含磁盘读取与硬件采集。不同记录可能共享物理轴承，公开数据准确率不能直接代表电钻效果。")
            with st.expander("模型结构与可追溯实验报告"):
                st.json({"model_config": manifest.get("model_config"), "selection": manifest.get("selection"), "data_protocol": manifest.get("data_protocol")})
                st.download_button("下载公开模型与实验清单", manifest_path.read_bytes(), "bearing_model_manifest.json", "application/json")
        except (ValueError, OSError) as error:
            st.error("模型报告无法读取：" + str(error))
    else:
        st.warning("公开数据模型报告尚未生成。")
        setup_commands()
    nasa_section()
    st.subheader("网页每部分怎么用")
    st.write("左侧选择实验页面。公开页面上方选记录、控制分析，中间看测试状态/预测/真实标签，下方看波形、特征与下载。虚拟页面上方设置生成条件，下方看三个传感器通道。当前页面用于理解算法与验证范围。")
    with st.expander("打开旧版仿真告警和维修工单"):
        st.write("旧版保留在 legacy_app.py。需要体验六维仿真健康、寿命和告警工单时，在另一终端启动；其模型和结果仍只来自原仿真。")
        st.code("python -m streamlit run legacy_app.py --server.port 8502", language="bash")
    st.subheader("现阶段还需要完成的验证")
    st.write("真实电动工具采集、跨工具/跨设备测试、自然退化与寿命标签、传感器缺失实验、独立维修复测以及目标 MCU 部署。任务书的准确率、寿命误差与提前预警天数仍是研发目标。")
    guide = ROOT / "docs" / "phase2" / "网页使用指南.md"
    if guide.exists():
        st.download_button("下载中文网页使用指南", guide.read_bytes(), "网页使用指南.md", "text/markdown")


st.title("电动工具实验与诊断平台")
st.caption("第二阶段研究平台 · 公开轴承故障预验证 + 虚拟多传感器实验 · 本网页不代表真实工具在线接入")
page = st.sidebar.radio("选择实验页面", PAGES)
st.sidebar.write("公开数据用于验证算法；虚拟平台用于设计实验。两个来源分别展示。")
st.sidebar.caption("新手先打开“公开轴承预验证”，点击开始分析，再到“模型与使用说明”了解结果适用范围。")
if page == PAGES[0]:
    public_page()
elif page == PAGES[1]:
    lab_page()
else:
    explanation_page()
