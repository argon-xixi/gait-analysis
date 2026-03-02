import os

def generate_gait_report_md(report_data, filename):
    """
    根据传入的报告数据，生成一个Markdown格式的步态分析报告。

    参数:
    report_data (dict): 包含报告所有信息的字典。
    filename (str): 要保存的.md文件名。
    """
    
    # --- 1. 从数据字典中提取信息 ---
    patient_info = report_data['patient_info']
    spatiotemporal_params = report_data['spatiotemporal_params']
    kinematic_params = report_data['kinematic_params']
    
    # --- 2. 动态生成参数表格的Markdown字符串 ---
    
    # 辅助函数：将数据列表转换成Markdown表格
    def create_md_table(headers, rows):
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join([":---"] * len(headers)) + " |"
        row_lines = []
        for row in rows:
            # 确保按表头的顺序提取值
            values = [str(row.get(key, '')) for key in headers]
            row_lines.append("| " + " | ".join(values) + " |")
        return "\n".join([header_line, separator_line] + row_lines)

    # 生成时空参数表格
    st_headers = ["参数", "左侧", "右侧", "平均值"]
    st_table = create_md_table(st_headers, spatiotemporal_params)
    
    # 生成运动学参数表格
    k_headers = ["参数", "左侧", "右侧", "平均值"]
    k_table = create_md_table(k_headers, kinematic_params)

    # --- 3. 定义Markdown模板并填充数据 ---
    # 使用 f-string 来填充模板，这使得代码非常清晰
    markdown_content = f"""
# 步态分析报告
### 步态分析报告
**报告编号：**{report_data.get('report_id', 'N/A')}
**报告时间：**{report_data.get('report_date', 'N/A')}

### 患者信息   

| **项目** | **详细信息** | 
| :-----| :---- | 
| **姓名** | {patient_info.get('姓名', '')} | 
| **性别** | {patient_info.get('性别', '')} | 
| **身高** | {patient_info.get('身高', '')} | 
| **体重** | {patient_info.get('体重', '')} | 
| **病史** | {patient_info.get('病史', '')} | 
| **测试目的** | {patient_info.get('测试目的', '')} | 

### 时空参数

{st_table}
 
### 运动学参数

{k_table}

### 图表可视化
#### 图1:各关节角度变化图
![各关节角度变化图]({report_data.get('image_path', '')})
"""

    # --- 4. 将生成的Markdown内容写入文件 ---
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Markdown报告 '{filename}' 已成功生成!")
    except IOError as e:
        print(f"错误：无法写入文件 {filename}。原因: {e}")

    return markdown_content

if __name__ == "__main__":
    
    # ==============================================================================
    #  核心修改：将所有可变参数定义在一个字典中，方便管理和修改
    # ==============================================================================
    report_data_zhangsan = {
        "report_id": "20250723-001",
        "report_date": "2025年7月23日",
        "patient_info": {
            "姓名": "张三",
            "性别": "男",
            "身高": "180cm",
            "体重": "70kg",
            "病史": "股骨干骨折术后3个月,AO分型B2(楔形骨折),术后内固定,当前康复阶段:术后中期(3周~3个月)",
            "测试目的": "评估术后步态恢复情况，监测跛行及稳定性，指导进一步康复训练"
        },
        "spatiotemporal_params": [
            {"参数": "步频(step/min)", "左侧": 96, "右侧": 96, "平均值": 96},
            {"参数": "步周期(s)", "左侧": 1.25, "右侧": 1.25, "平均值": 1.25},
            {"参数": "双支撑期(s)", "左侧": 0.23, "右侧": 0.23, "平均值": 0.23},
            {"参数": "支撑期比例(%)", "左侧": 69, "右侧": 72, "平均值": 70.5},
            {"参数": "摇摆期比例(%)", "左侧": 31, "右侧": 28, "平均值": 29.5}
        ],
        "kinematic_params": [
            {"参数": "膝关节角度(屈曲°)", "左侧": 64, "右侧": 67, "平均值": 65.5},
            {"参数": "踝关节角度(背屈°)", "左侧": 13, "右侧": 11, "平均值": 12},
            {"参数": "髋关节角度(伸展°)", "左侧": 38, "右侧": 40, "平均值": 39}
        ],
        # 重要提示：为了让Markdown文件能正确显示图片，推荐使用相对路径。
        # 例如，如果图片和生成的 .md 文件在同一个文件夹下，这里可以直接写 "00179.jpg"。
        # 绝对路径 "/home/yjh/..." 仅在您本地的Markdown查看器上可能有效。
        "image_path": "chart/00179.jpg" 
    }

    # 调用函数，传入数据和希望生成的文件名
    generate_gait_report_md(report_data_zhangsan, "/home/yjh/code_yjh/mmpose-main/yjh/步态分析报告-张三.md")
    
    # 示例：为另一位患者生成报告，只需准备新的数据即可
    report_data_lisi = report_data_zhangsan.copy() # 复制模板
    report_data_lisi["report_id"] = "20250724-002"
    report_data_lisi["patient_info"]["姓名"] = "李四"
    report_data_lisi["spatiotemporal_params"][0]["平均值"] = 102 # 修改李四的步频数据
    # generate_gait_report_md(report_data_lisi, "步态分析报告-李四.md")