from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "reference_style_reproduction"
GPT_DIR = ROOT / "outputs" / "gptimage2_diagrams"
OUT_CN = ROOT / "outputs" / "图表思路逐图解释与预算_NSCLC_MMFT.docx"
OUT_ASCII = ROOT / "outputs" / "figure_ideas_explained_budget_nsclc_mmft.docx"

FONT = "Microsoft YaHei"
INK = RGBColor(31, 41, 55)
BLUE = RGBColor(31, 78, 121)
RED = RGBColor(153, 27, 27)
GRAY = RGBColor(75, 85, 99)


FIGURE_ITEMS = [
    {
        "title": "Figure 1 | 模型性能总览：ROC、混淆矩阵、风险分布与关键指标",
        "path": FIG_DIR / "Figure1_reference_style_performance.png",
        "status": "真实公共 PET/CT-blood 测试集结果；本院外部验证位置先保留。",
        "goal": "这一张图回答论文最核心的问题：术前 PET/CT 代谢参数、血液/炎症指标和 CEA 融合后，能否比普通线性模型更好地区分 NSCLC 病理亚型。",
        "how_to_read": [
            "A 面板看 ROC 曲线和 AUC，曲线越靠左上角说明区分度越好；目前 MMFT rank-refit 的测试集 AUC 为 0.868。",
            "B 面板看交叉验证稳定性，如果不同折之间波动很大，说明小样本下模型不稳，需要谨慎解释。",
            "C 面板保留给本院验证队列，后期把 100 例左右的医院病例接入后，直接替换为外部验证 ROC。",
            "D/E 面板用混淆矩阵展示错分类型，临床上要重点关注假阴性，因为假阴性意味着模型漏掉高风险/目标类别病例。",
            "F/G 面板展示预测概率分布和阈值指标，能说明模型不是只报一个 AUC，而是有可落地的风险分层阈值。",
        ],
        "interpretation": [
            "公共数据中，MMFT rank-refit 的 AUC、准确率和 F1 均优于 Ridge logistic，提示融合风险 token 对小样本表格数据有增益。",
            "Ridge logistic 仍然作为强基线保留，这一点很重要：毕业论文不应只展示深度学习模型，还要证明深度学习确实比简单模型多提供了一点信息。",
            "目前这张图只能代表公共队列内部验证，不能写成临床泛化已经被证明；外部泛化需要本院队列补齐。",
        ],
        "writing": "可写作：在公共测试集中，MMFT rank-refit 获得最高区分度（AUC=0.868），较 Ridge logistic 呈现一定增益；后续本院队列将用于独立验证模型泛化能力。",
        "next_step": "本院数据收集完成后，把 C 面板替换为本院 ROC，并补充校准曲线和 DCA，论文可信度会明显提升。",
    },
    {
        "title": "Figure 2 | 亚组分析：模型在不同临床分层中的稳健性",
        "path": FIG_DIR / "Figure2_reference_style_subgroup_auc.png",
        "status": "真实公共测试集探索性亚组结果；bootstrap 置信区间来自当前样本。",
        "goal": "这一张图回答：模型是否只在某一类患者中表现好，还是在年龄、CEA、SUVmax、NLR、分期和性别等常见临床分层下都保持可接受表现。",
        "how_to_read": [
            "每个小图代表一个分层变量，横轴是亚组，纵轴是 AUC。",
            "蓝色/红色柱比较 Ridge 与 MMFT，误差线越宽说明该亚组样本越少或不确定性越大。",
            "例如 SUVmax 高低、NLR 高低和 CEA 高低亚组中，MMFT 多数保持 0.80 左右或以上的 AUC，说明模型没有完全依赖单一变量。",
            "Stage high 亚组样本量只有约 10 例，因此置信区间很宽，不能过度解读。",
        ],
        "interpretation": [
            "亚组分析的作用是增加模型可信度，而不是证明每个亚组均有统计学差异。",
            "小样本情况下，亚组图主要用于展示方向一致性；如果导师担心过拟合，可以把它写成探索性分析。",
            "该图也提示后续本院验证时要尽量保证关键亚组都有样本，例如 CEA 高低、NLR 高低、腺/鳞或 N0/N+。",
        ],
        "writing": "可写作：探索性亚组分析显示，MMFT 在多数临床分层中维持相对稳定的区分度，但样本量较小的亚组置信区间较宽，仍需外部队列验证。",
        "next_step": "本院队列若样本只有 100 例，不建议再切太多亚组；优先保留 3-4 个最有临床意义的亚组即可。",
    },
    {
        "title": "Figure 3 | 个体层面可解释性：SHAP 高低风险病例与风险分布",
        "path": FIG_DIR / "Figure3_reference_style_case_interpretability.png",
        "status": "真实模型输出和 SHAP 解释图；原参考文献的 Grad-CAM 在本项目中用 SHAP 替代。",
        "goal": "这一张图回答：模型为什么把某个患者判为高风险或低风险，哪些变量在推动预测结果。",
        "how_to_read": [
            "左/右两侧代表典型病例，条形向右说明该变量推高预测概率，向左说明降低预测概率。",
            "中间风险分布显示所有测试集病例的预测概率，观察两类病例是否被模型大致分开。",
            "当前 SHAP 重要性最高的是融合风险 token，其后是 SUVmin、CEA、SUVmean、性别和 SUVmax。",
            "这说明模型主要学习到 PET 代谢负荷、血液肿瘤标志物和融合表示，而不是依靠单一炎症指标硬分。",
        ],
        "interpretation": [
            "对于毕业论文，SHAP 的价值很高：它能把深度学习模型从黑箱变成可解释风险评分。",
            "该图可以和 Figure 8 的 TCGA 生信图形成呼应：CEA/CEACAM5、糖酵解代谢、鳞癌标志物共同支持模型变量的生物学合理性。",
            "但 SHAP 不能证明因果，只能解释模型在当前数据上的预测依据。",
        ],
        "writing": "可写作：SHAP 分析显示，融合风险 token、PET 代谢参数及 CEA 对预测贡献较大，提示模型可能捕捉了代谢活跃度与肿瘤分泌表型的联合信息。",
        "next_step": "后期可在本院 100 例上再跑一次 SHAP beeswarm，检查重要变量排序是否一致。",
    },
    {
        "title": "Figure 4 | 外部 TCGA 生存图：生物风险 proxy 与预后关系",
        "path": FIG_DIR / "Figure4_reference_style_km_tcga_real.png",
        "status": "真实 UCSC Xena/GDC TCGA-LUAD/LUSC survival 数据；不是模型外部验证，而是机制支持。",
        "goal": "这一张图回答：模型涉及的 CEA/代谢/鳞癌分化相关轴，在公共转录组数据中是否也与肿瘤生物学和预后存在关系。",
        "how_to_read": [
            "每个 KM 面板按 biological risk proxy 中位数分为高低组，纵轴是总体生存概率，横轴是随访时间。",
            "LUAD 中高低风险曲线分离更明显，当前 log-rank P 值约 0.0077，提示该 proxy 在肺腺癌中有较强预后关联。",
            "LUSC 和合并队列方向较弱或不显著，这并不是失败，而是提示不同病理亚型的生物轴可能不同。",
            "风险 proxy 由 CEACAM5、SLC2A1、HK2、LDHA、TP63、KRT5 等基因构成，分别对应 CEA 分泌、糖酵解和鳞癌标志物。",
        ],
        "interpretation": [
            "这张图不能写成“我们的模型可以预测生存”，因为当前模型输入不是转录组，公共 PET/CT 队列也没有随访时间。",
            "正确写法是：TCGA 外部数据提示模型关键变量背后存在可解释的生物学轴，为术前影像-血液模型提供间接机制支持。",
            "这种干实验很适合毕业论文：成本低、能补充新颖性，但必须界定为探索性验证。",
        ],
        "writing": "可写作：基于 TCGA-LUAD/LUSC 的外部干实验显示，CEA/糖酵解/鳞癌标志物构成的 biological risk proxy 在 LUAD 中与总体生存显著相关，支持模型变量具有一定生物学合理性。",
        "next_step": "如果需要更强，可以追加 GEO 队列方向验证，或加入 GSEA/GSVA 的糖酵解、缺氧、角化通路图。",
    },
    {
        "title": "Figure 5 | 临床研究路径：从术前资料到本院验证",
        "path": FIG_DIR / "Figure5_reference_style_clinical_pathway.png",
        "status": "根据本课题重新绘制的研究路径图。",
        "goal": "这一张图回答：模型在胸外科场景中怎么用，和医生已有流程有什么关系。",
        "how_to_read": [
            "左侧是术前可获得信息：PET/CT 或胸部 CT、血常规/炎症指标、CEA 和基础临床资料。",
            "中间是模型输出：不是替代病理，而是提供术前风险概率和可解释变量贡献。",
            "右侧是临床用途：辅助分层、提示是否需要更认真评估淋巴结或病理亚型，并作为本院验证的落点。",
            "图中强调 validation gate，意思是模型必须经过本院病例验证后，才能写成有临床应用潜力。",
        ],
        "interpretation": [
            "这张图让论文不只是算法竞赛，而是回到胸外科临床问题。",
            "若后续研究结局改为淋巴结转移或早期复发，这张图仍然可用，只需把输出端从 histology probability 改为 pN+ risk 或 recurrence risk。",
            "对于毕业，建议保守定位为“模型构建及外部验证预案”，不要写成实际指导治疗。",
        ],
        "writing": "可写作：本研究拟构建一条基于术前多模态资料的风险评估路径，将公共队列建模、TCGA 生物学论证与本院外部验证相结合。",
        "next_step": "本院病例收集时要固定术前抽血时间窗，例如术前 7 天或 14 天内最近一次血常规和 CEA。",
    },
    {
        "title": "Figure 6 | 纳排与队列流程：公共建模、干实验和本院验证",
        "path": FIG_DIR / "Figure6_reference_style_participant_flow.png",
        "status": "真实公共队列样本数 + 后续本院验证预案。",
        "goal": "这一张图回答：数据从哪里来、哪些病例被纳入、训练/测试/外部验证之间是什么关系。",
        "how_to_read": [
            "公共 PET/CT-blood 队列用于主模型构建和内部测试，当前 n=255。",
            "TCGA-LUAD/LUSC 分支用于生物信息学干实验，当前合并 n=1029。",
            "本院 100 例左右病例作为后续独立验证，不参与模型训练，以避免数据泄漏。",
            "如果本院只做验证，论文逻辑会更干净：公共库建模，本院验证，TCGA 解释机制。",
        ],
        "interpretation": [
            "流程图最容易被导师和评审检查，因此必须把训练集、测试集、外部验证和干实验分清楚。",
            "本院病例不能既调参又验证；如果样本很少，建议只做锁定模型后的外部测试。",
            "纳排标准建议写清：术前资料完整、病理明确、抽血时间窗明确、排除感染/血液系统疾病等明显影响炎症指标的情况。",
        ],
        "writing": "可写作：研究采用公共队列建模、公共转录组干实验论证和本院独立验证的三阶段设计，以提高小样本毕业课题的可执行性和方法学可信度。",
        "next_step": "收本院病例前，先把 Excel 字段模板定死，避免后面重新翻病历。",
    },
    {
        "title": "Figure 7 | GPTimage2 总流程图：论文 graphical workflow",
        "path": GPT_DIR / "Figure7_GPTimage2_research_workflow_final.png",
        "status": "GPTimage2 生成医学 AI workflow 底图，再由程序叠加准确标签；已替换为新版最终图。",
        "goal": "这一张图回答：整篇文章的技术路线是什么，适合作为图形摘要或方法学总览图。",
        "how_to_read": [
            "Data 层展示公共 NSCLC 队列、本院外部验证和 TCGA 生信数据三类来源。",
            "Model 层展示标准化、MMFT/rank-refit 模型、风险评分和 SHAP 解释。",
            "Analysis 层展示模型评价、亚组分析、TCGA 干实验和后续临床验证。",
            "图上文字由程序固定绘制，避免 AI 生图常见的小字拼写错误。",
        ],
        "interpretation": [
            "这张图的作用不是放具体结果，而是让读者一眼理解论文结构。",
            "AI 生图用于视觉底图可以，但正式投稿前最好把所有文字、箭头、节点用矢量软件或 Python 程序再固定一遍。",
            "毕业论文中可以放在方法开头，SCI 投稿中可作为 Graphical abstract 或 Figure 7。",
        ],
        "writing": "可写作：整体技术流程包括公共队列建模、可解释性分析、TCGA 外部生物学论证和本院独立验证四个模块。",
        "next_step": "如果导师要求更朴素，可以用 Figure7_reference_style_workflow_programmatic.png 替代这一张 AI 风格图。",
    },
    {
        "title": "Figure 8 | GPTimage2 模型架构图：MMFT 输入分支、Transformer 与输出评价",
        "path": GPT_DIR / "Figure9_GPTimage2_model_architecture_final.png",
        "status": "GPTimage2 生成深度学习架构底图，再由程序精确叠加模型模块标签。",
        "goal": "这一张图回答：我们自己设计的 MMFT 模型到底怎么把术前临床资料、血液炎症指标、PET/CT 代谢参数和可选 CT 语义特征融合起来。",
        "how_to_read": [
            "左侧四路输入分别是临床人口学、血液/CEA/炎症指标、PET/CT 代谢特征和可选 CT 语义或 radiomics 分支。",
            "每一路先做缺失值处理和标准化，再转换为 token，并进入 embedding 层。",
            "中间是 Transformer encoder，核心模块包括 cross-modal attention、feed-forward 和 residual 结构，用于学习不同模态之间的交互。",
            "底部 risk token 汇总患者层面的融合风险表示，右侧输出 rank-refit 校准、患者风险概率、ROC/AUC、校准/DCA 和 SHAP 解释。",
        ],
        "interpretation": [
            "这张图让深度学习部分不再只是口头描述，而是明确展示了输入、融合、输出和解释模块。",
            "对于小样本毕业课题，架构图要强调可复现和可解释：输入变量简单、模型有传统基线对照、输出可用 SHAP 解释。",
            "图中保留 optional CT branch，是为了兼容你后续本院只有胸部 CT、没有 PET/CT 的情况；如果最终不用 radiomics，可以把这一路写成 CT semantic signs。",
        ],
        "writing": "可写作：MMFT 将不同术前模态变量转换为 token 后输入 Transformer encoder，通过 cross-modal attention 学习模态间互补信息，并由 risk token 输出患者层面的预测概率；rank-refit 用于提高小样本排序稳定性，SHAP 用于解释变量贡献。",
        "next_step": "正式论文中需在方法部分说明每个输入分支的变量列表、缺失值处理、标准化方式、训练/验证划分和超参数范围。",
    },
    {
        "title": "Figure 9 | TCGA 外部生物信息学验证：差异、表达、通路 proxy 与 KM",
        "path": FIG_DIR / "Figure8_TCGA_external_bioinfo_validation.png",
        "status": "真实 TCGA-LUAD/LUSC expression + survival 数据生成。",
        "goal": "这一张图回答：为什么 PET 代谢、CEA 和病理亚型预测在生物学上说得通。",
        "how_to_read": [
            "A 面板类似 marker 差异散点图，展示 LUAD 与 LUSC 在关键基因上的表达差异方向。",
            "B 面板展示 CEACAM5、SLC2A1、HK2、LDHA、TP63、KRT5 等基因在 LUAD/LUSC 中的表达分布。",
            "C 面板是 pathway proxy 热图，把 CEA 分泌轴、糖酵解/缺氧轴、鳞癌分化轴做成可视化证据链。",
            "D 面板与 Figure 4 呼应，用 KM 说明 biological risk proxy 与预后之间的探索性关系。",
        ],
        "interpretation": [
            "这张图是本课题的“干实验新颖点”：不需要额外湿实验，却能把模型变量和肿瘤生物学联系起来。",
            "它不能代替临床外部验证，但能解释为什么 CEA、SUV 代谢参数和病理亚型之间有可能存在可学习信号。",
            "写作时要避免堆太多基因，保持 6-10 个核心基因最清楚。",
        ],
        "writing": "可写作：TCGA 外部转录组分析提示，CEA/CEACAM5、糖酵解相关基因及鳞癌分化标志物在 LUAD 与 LUSC 之间呈现差异表达，并可构成与预后相关的 biological risk proxy。",
        "next_step": "后续可补一张 GSEA 气泡图或 GSVA 箱线图，但不要把生信部分做得比主模型还复杂。",
    },
]


TABLE_ITEMS = [
    {
        "title": "Table 1 | 公共队列基线特征",
        "path": FIG_DIR / "Table1_baseline_characteristics_public_cohort.png",
        "status": "真实 PLOS PET/CT-blood 公共队列 baseline 数据，按 histology 0/1 分组。",
        "goal": "这一张表说明样本基本情况和候选变量范围，是方法学可信度的入口。",
        "how_to_read": [
            "连续变量用中位数和四分位数展示，适合小样本和偏态分布变量。",
            "表中包括年龄、BMI、WBC、NEU、LYM、PLT、CEA、NLR、SUVmean、SUVmax 等，和本课题的术前血液/炎症/PET 方向一致。",
            "Class 0/1 目前代表公共数据中的病理亚型标签，写论文时要在方法部分明确编码含义。",
        ],
        "interpretation": [
            "这张表证明公共库不是只有影像组学特征，还能纳入血液和 CEA 等较容易收集的临床变量。",
            "后续本院验证队列表要尽量复用同一变量格式，这样模型外部验证更顺。",
        ],
        "writing": "可写作：公共队列纳入 255 例 NSCLC 患者，术前临床、血液、炎症、CEA 及 PET 代谢变量均可用于模型构建。",
        "next_step": "补本院数据时，字段名尽量和这张表保持一致，例如 WBC、NEU、LYM、PLT、NLR、dNLR、CEA、SUVmax。",
    },
    {
        "title": "Table 2 | 公共测试集模型性能",
        "path": FIG_DIR / "Table2_model_performance_public_test.png",
        "status": "真实模型结果表。",
        "goal": "这张表把 ROC 图中的主要结果变成可检索数字，方便写摘要、结果和讨论。",
        "how_to_read": [
            "Ridge logistic 是传统机器学习基线，AUC=0.854。",
            "CV MMFT ensemble 的 AUC=0.830，说明初始深度模型并不一定优于强基线。",
            "MMFT rank-refit 的 AUC=0.868、Accuracy=0.792、Sensitivity=0.818、Specificity=0.758，为当前最佳结果。",
        ],
        "interpretation": [
            "这张表很关键：它诚实展示了并非所有 Transformer 版本都变好，最终改进来自小样本下更稳的 rank-refit 策略。",
            "讨论中可以写：深度学习在小样本表格数据上需要强正则、稳健验证和简单模型对照。",
        ],
        "writing": "可写作：与 Ridge logistic 和 CV MMFT ensemble 相比，MMFT rank-refit 在公共测试集中取得最高 AUC 和 F1，提示排序约束与后处理校准可改善小样本多模态建模稳定性。",
        "next_step": "本院验证时表格只需要新增一列 External validation，不建议重新训练一个复杂模型。",
    },
    {
        "title": "Table 3 | 消融实验与落地计划",
        "path": FIG_DIR / "Table3_ablation_and_landing_plan.png",
        "status": "真实已有结果 + 明确标注的后续计划。",
        "goal": "这张表解释模型为什么需要融合临床/血液、PET/CT 代谢、CEA 和风险 token。",
        "how_to_read": [
            "Full MMFT rank-refit 是当前最佳模型，AUC=0.868。",
            "Ridge risk token only 是强基线，AUC=0.854，说明简单模型已经很强。",
            "FT-Transformer、MMFT without rank loss 和 radiomics-heavy branch 是对照/既往运行结果，用于说明不是模型越复杂越好。",
            "Hospital external validation 标注为 pending，明确下一步工作，不把未完成内容写成结果。",
        ],
        "interpretation": [
            "消融表让论文更经得起推敲，因为它展示了每个模块的贡献和边界。",
            "如果导师希望毕业更稳，可以把深度学习表述为“增强模型”而不是唯一主模型，Ridge/Logistic 作为临床可解释基线。",
        ],
        "writing": "可写作：消融分析显示，融合模型较传统基线略有提升，但提升幅度有限，提示本研究应重点强调可解释性、外部验证和低成本可复现流程。",
        "next_step": "后续可以补校准曲线、DCA 和 VIF/共线性检查，增强统计方法学完整性。",
    },
]


BUDGET_ROWS = [
    [
        "公共 PET/CT-blood 主建模数据",
        "PLOS ONE / PMC",
        "必要",
        "0 元",
        "0.5-1 天",
        "在 PLOS 文章 Supporting information 下载 S1/S2 Dataset ZIP；本项目已完成下载和建模。",
        "公开文章提供 PET/CT radiomics features 与 baseline characteristics 数据。",
    ],
    [
        "TCGA-LUAD/LUSC 干实验",
        "NCI GDC + UCSC Xena",
        "必要",
        "0 元",
        "0.5-1 天",
        "从 UCSC Xena/GDC hub 下载 star_tpm 和 survival；本项目已生成 1029 例合并数据。",
        "用于 Figure 4 和 Figure 8 的外部生物学论证，不等同临床验证。",
    ],
    [
        "本院 100 例外部验证",
        "本院病案室/PACS/LIS",
        "必要",
        "通常 0 元；若医院收取数据导出/伦理材料费，按本院标准",
        "2-4 周",
        "先伦理备案或导师授权，再按固定 Excel 字段表导出术前 CT/PET、血常规、CEA 和病理标签。",
        "这是整篇文章最关键的补强；不建议把本院数据再用于调参。",
    ],
    [
        "CT/PET 标注与图像核对",
        "3D Slicer",
        "推荐",
        "0 元",
        "安装 0.5 天；100 例语义核对约 1-2 周",
        "从 3D Slicer 官网下载；两名医生独立标注，计算 Kappa/ICC，有争议由第三人仲裁。",
        "软件免费开源，可做影像浏览、分割和基础测量。",
    ],
    [
        "统计分析与机器学习",
        "R Project + Python + scikit-learn/PyTorch/SHAP",
        "必要",
        "0 元",
        "2-5 天复跑和整理",
        "使用开源环境；当前 scripts 文件夹已有建模、SHAP、TCGA 和绘图脚本。",
        "毕业论文建议保留 Logistic/Ridge 基线，深度模型作为增强模型。",
    ],
    [
        "GPU 训练",
        "崔老师服务器/自有服务器",
        "推荐",
        "新增 0 元",
        "1-2 天",
        "通过你提供的服务器信息登录，配置 Python 环境后运行训练脚本。",
        "本项目样本量小，显卡更多用于快速迭代，不是必须长期占用。",
    ],
    [
        "云 GPU 备用",
        "AutoDL 算力云",
        "可选",
        "RTX 4090 官网示例约 1.98 元/小时；24-72 小时约 48-143 元，预留 100-300 元更稳",
        "1-3 天",
        "AutoDL 官网注册/学生认证/充值，控制台选择按量计费、地区、GPU 型号和镜像，实例运行才计费，用完关机。",
        "适合服务器临时不可用时备份；可按消费或充值开票。",
    ],
    [
        "高质量统计图商业软件",
        "GraphPad Prism",
        "可选",
        "学生年费约 142 美元；月订阅约 50 美元；若用 Python 作图可 0 元",
        "购买后即用；免费试用 30 天",
        "GraphPad 官网 How to buy 页面购买；学生需按官网要求提交学生证明。",
        "不是刚需；本项目图已经可由 Python 复现。",
    ],
    [
        "英文润色",
        "意得辑 Editage",
        "可选",
        "标准润色约 0.42-0.50 元/词；6000 英文词约 2520-3000 元",
        "2-4 个工作日，急件另算",
        "Editage 官网上传稿件，选择标准润色和返稿时间，确认报价后付款。",
        "若只交中文毕业论文，可暂不购买；若投英文 SCI，建议预算保留。",
    ],
    [
        "英文润色备选",
        "AJE 美国期刊专家",
        "可选",
        "官网显示标准润色 314.35 元起，高级润色 1613.55 元起，VIP 3050.46 元起",
        "按服务等级和字数计算",
        "AJE 中文官网选择服务并在线下单。",
        "适合作为 Editage 的报价对照。",
    ],
    [
        "生信外包补图",
        "广州基迪奥生物科技有限公司 或 杭州联川生物",
        "不建议首选；可选",
        "官网通常需询价；建议只预留 3000-8000 元",
        "7-14 个工作日",
        "官网提交需求或联系客服，说明 TCGA/GEO 肺癌、差异分析、GSEA/GSVA、KM、免疫浸润图。",
        "当前我们已能自己完成 TCGA 图；除非导师要求公司报告，否则不建议花这笔钱。",
    ],
]


SOURCE_ROWS = [
    ["PLOS ONE 公共 PET/CT 数据", "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0300170"],
    ["PLOS PDF supporting information", "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0300170&type=printable"],
    ["NCI TCGA 项目", "https://www.cancer.gov/ccg/research/genome-sequencing/tcga"],
    ["UCSC Xena / GDC hub", "https://xena.ucsc.edu/"],
    ["3D Slicer", "https://www.slicer.org/"],
    ["R Project", "https://www.r-project.org/about.html"],
    ["AutoDL", "https://www.autodl.com/"],
    ["GraphPad Prism pricing", "https://www.graphpad.com/how-to-buy/"],
    ["Editage pricing", "https://www.editage.cn/pricing/editing-service"],
    ["AJE pricing", "https://www.aje.cn/"],
    ["基迪奥生物", "https://www.genedenovo.com/"],
    ["联川生物", "https://www.lc-bio.com/"],
]


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_run_font(run, size: float | None = None, bold: bool = False, color: RGBColor | None = None) -> None:
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def style_paragraph(paragraph, size: float = 9, color: RGBColor = INK, bold: bool = False) -> None:
    for run in paragraph.runs:
        set_run_font(run, size=size, bold=bold, color=color)


def set_landscape(section) -> None:
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(0.42)
    section.bottom_margin = Inches(0.42)
    section.left_margin = Inches(0.48)
    section.right_margin = Inches(0.48)


def add_title(doc: Document, text: str, subtitle: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    set_run_font(r, 20, True, BLUE)
    p.space_after = Pt(2)

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run(subtitle)
    set_run_font(r2, 10, False, GRAY)
    p2.space_after = Pt(10)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_run_font(r, 15 if level == 1 else 12, True, BLUE)
    p.space_before = Pt(6)
    p.space_after = Pt(4)


def add_body(doc: Document, text: str, color: RGBColor = INK, size: float = 9.5) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_run_font(r, size, False, color)
    p.paragraph_format.line_spacing = 1.06
    p.space_after = Pt(3)


def add_small_note(doc: Document, text: str, color: RGBColor = GRAY) -> None:
    add_body(doc, text, color=color, size=8.5)


def image_width_for_page(path: Path, max_width: float = 8.9, max_height: float = 3.95) -> float:
    try:
        with Image.open(path) as im:
            ratio = im.height / im.width
        return max(5.8, min(max_width, max_height / ratio))
    except Exception:
        return max_width


def add_key_value_table(doc: Document, rows: list[tuple[str, str | list[str]]]) -> None:
    table = doc.add_table(rows=0, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for label, value in rows:
        row = table.add_row()
        row.cells[0].width = Inches(1.45)
        row.cells[1].width = Inches(9.05)
        row.cells[0].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        row.cells[1].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        set_cell_shading(row.cells[0], "EAF1FE")
        p0 = row.cells[0].paragraphs[0]
        p0.text = label
        style_paragraph(p0, 8.5, BLUE, True)
        if isinstance(value, list):
            row.cells[1].text = ""
            for item in value:
                p = row.cells[1].add_paragraph(style=None)
                p.paragraph_format.left_indent = Inches(0.08)
                p.paragraph_format.first_line_indent = Inches(-0.08)
                r = p.add_run("• " + item)
                set_run_font(r, 8.3, False, INK)
                p.space_after = Pt(1)
        else:
            p1 = row.cells[1].paragraphs[0]
            p1.text = value
            style_paragraph(p1, 8.3, INK, False)
    doc.add_paragraph().space_after = Pt(1)


def add_figure_page(doc: Document, item: dict) -> None:
    doc.add_page_break()
    add_heading(doc, item["title"], level=2)
    add_small_note(doc, "数据状态：" + item["status"], color=RED if "保留" in item["status"] or "计划" in item["status"] else GRAY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if item["path"].exists():
        p.add_run().add_picture(str(item["path"]), width=Inches(image_width_for_page(item["path"])))
    else:
        r = p.add_run(f"缺失图片：{item['path']}")
        set_run_font(r, 10, True, RED)
    p.space_after = Pt(4)

    add_key_value_table(
        doc,
        [
            ("这图回答", item["goal"]),
            ("怎么看", item["how_to_read"]),
            ("怎么解释", item["interpretation"]),
            ("可写句式", item["writing"]),
            ("后续补强", item["next_step"]),
        ],
    )


def add_budget_table(doc: Document) -> None:
    add_heading(doc, "预算表：按毕业最低成本优先设计", level=1)
    add_body(
        doc,
        "结论先放前面：这篇文章最省钱的做法是公共库建模 + TCGA 干实验 + 本院 100 例独立验证。"
        "必要软件和公共数据库基本 0 元；真正需要花钱的通常只有云 GPU 备用、英文润色或导师要求的生信外包。",
    )

    headers = ["模块", "公司/平台", "必要性", "预计费用", "预计耗时", "怎么购买/使用", "备注"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for idx, h in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = h
        set_cell_shading(cell, "1F4E79")
        for p in cell.paragraphs:
            style_paragraph(p, 7.2, RGBColor(255, 255, 255), True)

    for values in BUDGET_ROWS:
        cells = table.add_row().cells
        for i, value in enumerate(values):
            cells[i].text = value
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            for p in cells[i].paragraphs:
                style_paragraph(p, 6.7, INK, False)
        if values[2] == "必要":
            set_cell_shading(cells[2], "E2F0D9")
        elif values[2] == "推荐":
            set_cell_shading(cells[2], "FFF2CC")
        else:
            set_cell_shading(cells[2], "FCE4D6")

    add_heading(doc, "推荐购买路线", level=2)
    add_key_value_table(
        doc,
        [
            (
                "最低成本版",
                "不购买 GraphPad、不外包生信、不租云 GPU；用现有服务器和 Python/R 完成。费用约 0 元，主要成本是 4-6 周人工收集和核对本院数据。",
            ),
            (
                "稳妥毕业版",
                "保留 AutoDL 100-300 元备用预算；如果英文投稿，Editage 标准润色按 6000 词约 2520-3000 元准备。总预算约 3000 元以内。",
            ),
            (
                "增强投稿版",
                "如导师要求更精美图或公司报告，再联系基迪奥/联川询价做 TCGA/GEO 图补强，预留 3000-8000 元；但当前阶段不建议优先购买。",
            ),
        ],
    )


def add_source_table(doc: Document) -> None:
    add_heading(doc, "预算与数据来源链接", level=1)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(["来源", "链接"]):
        cell = table.rows[0].cells[i]
        cell.text = h
        set_cell_shading(cell, "D9EAF7")
        for p in cell.paragraphs:
            style_paragraph(p, 8, BLUE, True)
    for name, url in SOURCE_ROWS:
        cells = table.add_row().cells
        cells[0].text = name
        cells[1].text = url
        for cell in cells:
            for p in cell.paragraphs:
                style_paragraph(p, 7.2, INK, False)


def add_overview(doc: Document) -> None:
    add_heading(doc, "一、整篇论文主线", level=1)
    add_key_value_table(
        doc,
        [
            (
                "推荐题目",
                "基于术前 PET/CT 代谢特征与血液炎症指标的 NSCLC 病理亚型多模态预测模型构建及 TCGA 生物信息学论证",
            ),
            (
                "毕业定位",
                "公共数据库先完成主模型，本院约 100 例后期作为独立验证；重点不是追求极复杂深度学习，而是把数据来源、模型评价、解释性和干实验链条做完整。",
            ),
            (
                "核心结果",
                "公共测试集中 MMFT rank-refit 当前 AUC=0.868，优于 Ridge logistic AUC=0.854；SHAP 显示融合风险 token、SUVmin、CEA、SUVmean 等贡献较大。",
            ),
            (
                "生信补强",
                "TCGA-LUAD/LUSC 真实数据 n=1029，用 CEACAM5、SLC2A1、HK2、LDHA、TP63、KRT5 构建 biological risk proxy，作为模型变量的外部生物学支持。",
            ),
            (
                "最重要限制",
                "公共 PET/CT-blood 队列当前结局是病理亚型；如果最终要写胸外科淋巴结转移或早期复发，本院数据必须提供对应金标准标签，不能用 TCGA 或公共亚型标签替代。",
            ),
        ],
    )

    add_heading(doc, "二、图表对应关系", level=1)
    rows = [
        ["Fig.1", "模型性能总览", "ROC、交叉验证、混淆矩阵、风险分布", "真实公共测试集 + 本院验证预留"],
        ["Fig.2", "亚组分析", "年龄、CEA、SUVmax、NLR、分期、性别", "真实探索性亚组"],
        ["Fig.3", "解释性", "SHAP 个体解释 + 风险分布", "真实模型输出"],
        ["Fig.4", "生存曲线", "TCGA biological risk proxy KM", "真实 TCGA survival"],
        ["Fig.5", "临床路径", "术前资料 -> 模型 -> 外部验证", "研究设计图"],
        ["Fig.6", "纳排流程", "公共建模、TCGA、本院验证", "真实样本数 + 计划"],
        ["Fig.7", "总流程图", "GPTimage2 底图 + 程序准确标注", "AI 视觉图"],
        ["Fig.8", "模型架构图", "MMFT 输入分支、Transformer、risk token、SHAP", "AI 底图 + 程序准确标注"],
        ["Fig.9", "干实验", "TCGA 差异、表达、通路 proxy、KM", "真实 TCGA expression/survival"],
        ["Table1", "基线表", "公共队列临床/血液/PET 变量", "真实数据"],
        ["Table2", "性能表", "Ridge、CV MMFT、MMFT rank-refit", "真实结果"],
        ["Table3", "消融表", "模块贡献与本院验证计划", "真实结果 + pending 标注"],
    ]
    table = doc.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(["图表", "用途", "主要内容", "数据状态"]):
        table.rows[0].cells[i].text = h
        set_cell_shading(table.rows[0].cells[i], "D9EAF7")
        for p in table.rows[0].cells[i].paragraphs:
            style_paragraph(p, 8, BLUE, True)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = value
            for p in cells[i].paragraphs:
                style_paragraph(p, 7.5, INK, False)


def main() -> None:
    doc = Document()
    set_landscape(doc.sections[0])
    doc.styles["Normal"].font.name = FONT
    doc.styles["Normal"].font.size = Pt(9)

    add_title(
        doc,
        "NSCLC-MMFT 图表思路逐图解释与预算表",
        "公共 PET/CT-blood 建模 + TCGA 干实验 + 本院外部验证预案",
    )
    add_overview(doc)

    for item in FIGURE_ITEMS:
        add_figure_page(doc, item)
    for item in TABLE_ITEMS:
        add_figure_page(doc, item)

    doc.add_page_break()
    add_budget_table(doc)
    doc.add_page_break()
    add_source_table(doc)

    doc.save(OUT_CN)
    copyfile(OUT_CN, OUT_ASCII)
    print(OUT_CN)
    print(OUT_ASCII)


if __name__ == "__main__":
    main()
