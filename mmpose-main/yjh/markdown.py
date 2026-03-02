import markdown2
def generate_markdown(data,filename):
    markdown_text = markdown2.markdown(data)
    with open(filename,"w",encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"{filename}文件已生成!")
    return markdown_text

if __name__=="__main__":
    content="""
#步态分析报告
###步态分析报告
**报告编号：**20250723-001
**报告时间：**2025年7月23日

###患者信息   

| **项目** | **详细信息** | 
| :-----| :---- | 
| **姓名** | 张三 | 
| **性别** | 男 | 
| **身高** | 180cm | 
| **体重** | 70kg | 
| **病史** | 股⻣⼲⻣折术后3个⽉,AO分型B2(楔形⻣折),术后内固定,当前康复阶段:术后中期(3周~3个⽉) | 
| **测试目的：** | 评估术后步态恢复情况，监测跛⾏及稳定性，指导进⼀步康复训练 | 

###时空参数

| **参数 | **左侧** |**右侧** | **平均值** | 
| :-----| :---- |  :-----| :---- |
| **步频(step/min)** | 96 | 96 | 96 |
| **步周期(s)** | 1.25 | 1.25 | 1.25 |
| **双支撑期(s)** | 0.23 | 0.23 | 0.23 | 
| **支撑期比例(%)** | 69 | 72 | 70.5 |
| **摇摆期比例(%)** | 31 | 28 | 29.5 |
 
###运动学参数
| **参数 | **左侧** |**右侧** | **平均值** | 
| :-----| :---- |  :-----| :---- |
| **膝关节角度(屈曲°)*** | 64 | 67 | 65.5 |
| **踝关节角度(背屈°)** | 13 | 11 | 12 |
| **髋关节角度(伸展°)** | 38 | 40 | 39 | 

###图表可视化
####图1:各关节角度变化图
    <img src="/home/yjh/code_yjh/mmpose-main/yjh/ori/walk_woman_processed_2/chart/00179.jpg" style="float:left;width:80%;" />
    """
    generate_markdown(content,"/home/yjh/code_yjh/mmpose-main/yjh/test.md")