#!/usr/bin/env python3
"""
Compiles CS 557 HW2 Analysis & Results into a publication-grade PDF report.
Includes complete coverage of Part 1A (REINFORCE), Part 1B (PPO), Part 1C BONUS (SAC),
and Part 2 Hyperparameter Ablations & Tuning.
Compiles CS 557 HW2 Analysis & Results into a publication-grade PDF report (Tri_Vo_hw2.pdf).
Uses bundled DejaVu TrueType fonts for clean Unicode rendering (zero missing-glyph boxes).
Author: Tri Vo (vo43@purdue.edu)
Purdue University - CS 557 RLI Fall 2026
"""

import os
import glob
import matplotlib
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILENAME = os.path.join(BASE_DIR, "Tri_Vo_hw2.pdf")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def register_fonts():
    ttf_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(ttf_dir, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(ttf_dir, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", os.path.join(ttf_dir, "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-BoldOblique", os.path.join(ttf_dir, "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSansMono", os.path.join(ttf_dir, "DejaVuSansMono.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSansMono-Bold", os.path.join(ttf_dir, "DejaVuSansMono-Bold.ttf")))
    pdfmetrics.registerFontFamily(
        "DejaVuSans",
        normal="DejaVuSans",
        bold="DejaVuSans-Bold",
        italic="DejaVuSans-Oblique",
        boldItalic="DejaVuSans-BoldOblique",
    )
    pdfmetrics.registerFontFamily(
        "DejaVuSansMono",
        normal="DejaVuSansMono",
        bold="DejaVuSansMono-Bold",
    )


def fit_image(filename, max_width=528, max_height=290):
    path = os.path.join(PLOTS_DIR, filename)
    with PILImage.open(path) as im:
        w, h = im.size
    aspect = h / float(w)
    draw_w = max_width
    draw_h = draw_w * aspect
    if draw_h > max_height:
        draw_h = max_height
        draw_w = draw_h / aspect
    return Image(path, width=draw_w, height=draw_h)


def build_pdf():
    register_fonts()

    doc = SimpleDocTemplate(
        PDF_FILENAME,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
        bottomMargin=36,
        title="CS 557 Fall 2026 Homework 2 Report - Tri Vo",
        author="Tri Vo",
    )

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#002D62")
    gold = colors.HexColor("#8E6500")
    slate = colors.HexColor("#1E242B")
    muted = colors.HexColor("#555E68")

    # Custom typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#002D62'), # Purdue Navy
        alignment=1, # Center
        spaceAfter=4
        "DocTitle", parent=styles["Normal"], fontName="DejaVuSans-Bold",
        fontSize=16.5, leading=21, textColor=navy, alignment=1, spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#555555'),
        alignment=1,
        spaceAfter=10
        "DocSubtitle", parent=styles["Normal"], fontName="DejaVuSans",
        fontSize=9.5, leading=13, textColor=muted, alignment=1, spaceAfter=10,
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor('#002D62'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
        "H1Custom", parent=styles["Normal"], fontName="DejaVuSans-Bold",
        fontSize=12.5, leading=16, textColor=navy, spaceBefore=10, spaceAfter=4, keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#C28E0E'), # Purdue Gold
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
        "H2Custom", parent=styles["Normal"], fontName="DejaVuSans-Bold",
        fontSize=10.5, leading=14, textColor=gold, spaceBefore=7, spaceAfter=3, keepWithNext=True,
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1E1E24'),
        spaceAfter=5
        "BodyCustom", parent=styles["Normal"], fontName="DejaVuSans",
        fontSize=9.0, leading=13.0, textColor=slate, spaceAfter=4,
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-9,
        spaceAfter=3
        "BulletCustom", parent=body_style, leftIndent=14, firstLineIndent=-9, spaceAfter=3,
    )

    code_box = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#111111'),
        backColor=colors.HexColor('#F5F6F8'),
        spaceBefore=3,
        spaceAfter=5,
        leftIndent=8,
        rightIndent=8
    caption_style = ParagraphStyle(
        "CaptionCustom", parent=styles["Normal"], fontName="DejaVuSans-Oblique",
        fontSize=8.0, leading=11.0, textColor=muted, alignment=1, spaceBefore=2, spaceAfter=8,
    )
    th_style = ParagraphStyle(
        "THCustom", parent=styles["Normal"], fontName="DejaVuSans-Bold",
        fontSize=7.8, leading=10.0, textColor=colors.white, alignment=1,
    )
    td_style = ParagraphStyle(
        "TDCustom", parent=styles["Normal"], fontName="DejaVuSans",
        fontSize=7.6, leading=9.8, textColor=slate, alignment=1,
    )
    td_left = ParagraphStyle("TDLeft", parent=td_style, alignment=0)

    def make_table(headers, rows, col_widths):
        data = [[Paragraph(h, th_style) for h in headers]]
        for r in rows:
            row_cells = []
            for idx, val in enumerate(r):
                row_cells.append(Paragraph(str(val), td_left if idx == 0 else td_style))
            data.append(row_cells)
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F9")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD3DC")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    story = []

    # Title Banner
    story.append(Paragraph("CS 557 Reinforcement Learning — Homework Assignment 2", title_style))
    story.append(Paragraph("<b>Author:</b> Tri Vo &nbsp;|&nbsp; <b>Email:</b> vo43@purdue.edu &nbsp;|&nbsp; <b>Purdue University</b> &nbsp;|&nbsp; Fall 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#C28E0E'), spaceAfter=8))
    # Header
    story.append(Paragraph("CS 557 Fall 2026 — Homework 2: Policy Gradient & Actor-Critic Methods", title_style))
    story.append(Paragraph(
        "Student: <b>Tri Vo</b> (vo43@purdue.edu) &nbsp;|&nbsp; Purdue University &nbsp;|&nbsp; "
        "Environments: <i>MountainCarContinuous-v0</i> &amp; <i>BipedalWalker-v3</i>",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=navy, spaceAfter=8))

    # Executive Summary
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Paragraph("1. Executive Summary & Success Threshold Audit", h1_style))
    story.append(Paragraph(
        "This report provides a rigorous empirical and mathematical evaluation of continuous policy gradient and actor-critic algorithms: "
        "<b>REINFORCE</b> (Monte Carlo Policy Gradient), <b>Proximal Policy Optimization (PPO)</b> (Clipped On-Policy Actor-Critic), "
        "and <b>Soft Actor-Critic (SAC)</b> (Off-Policy Maximum Entropy Actor-Critic, <b>Bonus Question</b>). "
        "All algorithms were implemented from first principles, tested locally for numerical validity, and executed on the Purdue RCAC Scholar high-performance compute cluster across "
        "<code>MountainCarContinuous-v0</code> and <code>BipedalWalker-v3</code>.",
        body_style
        "All three continuous-control algorithms — <b>REINFORCE</b> (Part 1A), <b>Proximal Policy Optimization (PPO)</b> "
        "(Part 1B &amp; Part 2), and <b>Soft Actor-Critic (SAC)</b> (Part 1C Bonus) — were implemented, mathematically "
        "verified on unit and Andy Jones probe tests (<font name='DejaVuSansMono'>test_agents.py</font>), and evaluated "
        "without modifying Gymnasium reward functions. Every solvable task passes its official Gymnasium threshold on "
        "both TensorBoard checkpoint evaluation and held-out 20-episode (and 100-episode) deterministic evaluation:",
        body_style,
    ))

    # Part 1: Implement Agents
    story.append(Paragraph("Part 1: Implement Agents", h1_style))
    exec_headers = [
        "Part & Algorithm", "Environment", "Steps", "Train Last-10", "TB Final Eval", "Held-Out Eval (20 eps)", "Threshold / Status"
    ]
    exec_rows = [
        ["1A: REINFORCE", "MountainCarContinuous-v0", "200,764", "-31.49 ± 1.28", "-1.80", "-1.80 ± 0.01", "≥ +90 (Expected Fail)"],
        ["1B: PPO + gSDE", "MountainCarContinuous-v0", "200,047", "+95.07 ± 0.28", "+95.18", "<b>+95.17 ± 0.17</b>", "≥ +90: <b>PASS</b>"],
        ["1C: SAC + OU (Bonus)", "MountainCarContinuous-v0", "200,061", "+94.94 ± 0.27", "+94.73", "<b>+94.98 ± 0.33</b>", "≥ +90: <b>PASS</b>"],
        ["1A: REINFORCE", "BipedalWalker-v3", "201,432", "-85.86 ± 30.36", "-113.79", "-118.11 ± 18.92", "≥ +300 (Expected Fail)"],
        ["1B: PPO Baseline", "BipedalWalker-v3", "200,424", "+39.97 ± 17.30", "-109.55", "-109.56 ± 0.13", "≥ +300 (Baseline)"],
        ["1C: SAC Tuned (Bonus)", "BipedalWalker-v3", "500,509", "+294.06 ± 87.43", "+328.86", "<b>+328.79 ± 0.70</b> (20/20)<br/>100-ep: 326.64 ± 21.98", "≥ +300: <b>PASS</b>"],
        ["2B: PPO Tuned", "BipedalWalker-v3", "2,500,034", "+306.50 ± 1.79", "+306.66", "<b>+307.75 ± 1.48</b> (20/20)<br/>100-ep: 301.54 ± 44.63", "≥ +300: <b>PASS</b>"],
    ]
    story.append(make_table(exec_headers, exec_rows, [76, 118, 52, 76, 44, 102, 72]))
    story.append(Spacer(1, 6))

    # Part 1A: REINFORCE
    story.append(Paragraph("Part 1A: REINFORCE (Monte Carlo Policy Gradient) [2 pts]", h2_style))
    story.append(Paragraph(
        "<b>Mathematical Formulation:</b> The REINFORCE algorithm optimizes expected discounted return J(&theta;) = E<sub>&tau;~&pi;<sub>&theta;</sub></sub> [R(&tau;)] via the Policy Gradient Theorem: "
        "&nabla;<sub>&theta;</sub> J(&theta;) = E<sub>&tau;~&pi;<sub>&theta;</sub></sub> [&sum;<sub>t=0</sub><sup>T-1</sup> &nabla;<sub>&theta;</sub> log &pi;<sub>&theta;</sub>(a<sub>t</sub>|s<sub>t</sub>) G<sub>t</sub>]. "
        "Here, G<sub>t</sub> = &sum;<sub>k=t</sub><sup>T-1</sup> &gamma;<sup>k-t</sup> r<sub>k</sub> is the discounted return accumulated backwards via dynamic programming. "
        "To curtail gradient variance without introducing bias, empirical returns are standardized: G&#770;<sub>t</sub> = (G<sub>t</sub> - &mu;<sub>G</sub>) / (&sigma;<sub>G</sub> + 1e-8).<br/>"
        "<b>Implementation Architecture:</b> In <code>reinforce.py</code>, <code>select_action</code> predicts state-conditioned Gaussian mean &mu;<sub>&theta;</sub>(s) with learnable log standard deviation log &sigma; (clamped to [-5.0, 0.0]). "
        "Actions are sampled u<sub>t</sub> ~ N(&mu;, &sigma;<sup>2</sup>), mapped to environment bounds via affine transformation, and the joint log-likelihood is accumulated over all action dimensions.",
        body_style
    ))
    story.append(Paragraph("Key Implementation Fixes", h2_style))
    story.append(Paragraph("• <b>SAC Truncation vs. Termination (<font name='DejaVuSansMono'>sac.py</font>):</b> Stored <font name='DejaVuSansMono'>float(terminated)</font> instead of <font name='DejaVuSansMono'>terminated or truncated</font> in the NumPy ring buffer so 999-step / 1,600-step time limits do not zero out future Q-targets.", bullet_style))
    story.append(Paragraph("• <b>SAC Temperature Gradient Isolation (<font name='DejaVuSansMono'>sac.py</font>):</b> Detached <font name='DejaVuSansMono'>self.alpha</font> in <font name='DejaVuSansMono'>policy_loss</font> and recomputed <font name='DejaVuSansMono'>self.log_alpha.exp()</font> inside <font name='DejaVuSansMono'>alpha_loss</font>.", bullet_style))
    story.append(Paragraph("• <b>Temporally Correlated Exploration (Unmodified Rewards):</b> Added optional gSDE + buffered <font name='DejaVuSansMono'>RunningObsNorm</font> in <font name='DejaVuSansMono'>ppo.py</font> and Ornstein-Uhlenbeck warmup in <font name='DejaVuSansMono'>sac.py</font> to solve <i>MountainCarContinuous-v0</i> cleanly without any reward shaping.", bullet_style))
    story.append(Paragraph("• <b>Collision-Free Logging &amp; Exact Ablation Parsing:</b> Exposed <font name='DejaVuSansMono'>logger.run_name</font> for unique checkpoints and replaced prefix matching with exact hyperparameter token matching in <font name='DejaVuSansMono'>plot_and_analyze.py</font>.", bullet_style))

    # Part 1B: PPO
    story.append(Paragraph("Part 1B: Proximal Policy Optimization (PPO) [4 pts]", h2_style))
    # Part 1A
    story.append(PageBreak())
    story.append(Paragraph("2. Part 1A: REINFORCE (Monte-Carlo Policy Gradient)", h1_style))
    story.append(Paragraph(
        "<b>Mathematical Formulation:</b> PPO maximizes a pessimistic surrogate objective bounded by a trust region: "
        "L<sup>CLIP</sup>(&theta;) = E&#770;<sub>t</sub> [min(r<sub>t</sub>(&theta;)A&#770;<sub>t</sub>, clip(r<sub>t</sub>(&theta;), 1-&epsilon;, 1+&epsilon;)A&#770;<sub>t</sub>)], "
        "where r<sub>t</sub>(&theta;) = exp(log &pi;<sub>&theta;</sub>(a<sub>t</sub>|s<sub>t</sub>) - log &pi;<sub>old</sub>(a<sub>t</sub>|s<sub>t</sub>)). "
        "Advantages A&#770;<sub>t</sub> are computed using Generalized Advantage Estimation (GAE): &delta;<sub>t</sub><sup>V</sup> = r<sub>t</sub> + &gamma; V<sub>&phi;</sub>(s<sub>t+1</sub>)(1-d<sub>t</sub>) - V<sub>&phi;</sub>(s<sub>t</sub>), "
        "and A&#770;<sub>t</sub> = &delta;<sub>t</sub><sup>V</sup> + &gamma;&lambda;(1-d<sub>t</sub>)A&#770;<sub>t+1</sub>.<br/>"
        "<b>Implementation Architecture:</b> In <code>ppo.py</code>, <code>compute_gae</code> computes advantages and Monte Carlo targets R<sub>t</sub> = r<sub>t</sub> + &gamma;(1-d<sub>t</sub>)R<sub>t+1</sub>. "
        "In <code>update_policy</code>, mini-batch updates are executed over 10 epochs with Adam (&alpha;=5e-5), combining clipped actor loss, critic MSE loss 0.5(V(s) - R)<sup>2</sup>, and entropy bonus c<sub>2</sub> H(&pi;).",
        body_style
        "<b>Implementation:</b> <font name='DejaVuSansMono'>REINFORCEAgent</font> uses a 2-layer 256-unit ReLU policy network "
        "predicting action mean μ(s) with a learnable <font name='DejaVuSansMono'>log_std</font> clamped to [-5.0, 0.0]. "
        "Full-episode discounted returns G<sub>t</sub> = ∑ γ<sup>k</sup> r<sub>t+k</sub> are standardized per episode and "
        "used to minimize -E[log π(a<sub>t</sub>|s<sub>t</sub>) G̃<sub>t</sub>].",
        body_style,
    ))

    # Part 1C: BONUS Soft Actor-Critic (SAC)
    story.append(Paragraph("Part 1C (BONUS): Soft Actor-Critic (SAC) [2 pts]", h2_style))
    story.append(Paragraph(
        "<b>Maximum Entropy Reinforcement Learning:</b> SAC maximizes return augmented with policy entropy: "
        "J(&pi;) = &sum;<sub>t=0</sub><sup>T</sup> E<sub>(s<sub>t</sub>, a<sub>t</sub>)</sub> [r(s<sub>t</sub>, a<sub>t</sub>) + &alpha; H(&pi;(&middot;|s<sub>t</sub>))], "
        "where H(&pi;) = -E [log &pi;(a|s)] encourages broad exploration and prevents policy collapse into suboptimal deterministic extrema.<br/>"
        "<b>Twin Critic Networks & Clipped Double Q-Learning:</b> To eliminate overestimation bias in continuous action domains, SAC maintains two critics Q<sub>&phi;<sub>1</sub></sub>, Q<sub>&phi;<sub>2</sub></sub> "
        "and exponential moving average target critics Q<sub>&phi;&#772;<sub>1</sub></sub>, Q<sub>&phi;&#772;<sub>2</sub></sub> (&tau;=0.005). The soft Bellman target is:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;y = r + &gamma;(1 - d) [ min<sub>i=1,2</sub> Q<sub>&phi;&#772;<sub>i</sub></sub>(s', a&#771;') - &alpha; log &pi;<sub>&theta;</sub>(a&#771;'|s') ], &nbsp;&nbsp; a&#771;' ~ &pi;<sub>&theta;</sub>(&middot;|s').<br/>"
        "Critic parameters minimize Mean Squared Bellman Error: L<sub>critic</sub> = E [(Q<sub>1</sub>(s,a) - y)<sup>2</sup> + (Q<sub>2</sub>(s,a) - y)<sup>2</sup>].<br/>"
        "<b>Reparameterization Trick & Tanh Squashing:</b> To enable differentiable policy optimization through continuous sampling, actions are sampled via the reparameterization trick: "
        "u = &mu;<sub>&theta;</sub>(s) + &sigma;<sub>&theta;</sub>(s) &odot; &xi;, with &xi; ~ N(0, I), and mapped via a = tanh(u) &in; (-1, 1). "
        "The exact action density accounts for the invertible squash mapping using the Jacobian change-of-variables correction: "
        "log &pi;<sub>&theta;</sub>(a|s) = log &mu;(u|s) - &sum;<sub>i=1</sub><sup>D</sup> log(1 - tanh<sup>2</sup>(u<sub>i</sub>) + 1e-6).<br/>"
        "<b>Policy Objective & Dual Temperature Optimization:</b> The actor minimizes L<sub>actor</sub> = E<sub>s~D, a&#771;~&pi;</sub> [&alpha; log &pi;<sub>&theta;</sub>(a&#771;|s) - min<sub>i=1,2</sub> Q<sub>i</sub>(s, a&#771;)]. "
        "Temperature &alpha; is adaptively tuned to maintain target entropy H&#772; = -dim(A): L(&alpha;) = E [-&alpha; (log &pi;(a|s) + H&#772;)].<br/>"
        "<b>Code Completion in <code>sac.py</code>:</b><br/>"
        "1. <code>PolicyNetwork.sample</code>: Implemented Gaussian reparameterization via <code>normal.rsample()</code>, computed squashed actions via <code>torch.tanh(u)</code>, and applied exact log-sum Jacobian correction.<br/>"
        "2. <code>SACAgent.update_parameters</code>: Implemented double Q target calculation with entropy subtraction, MSE critic regression, actor loss minimization against min-Q, temperature parameter update, and Polyak target updates.",
        body_style
        "<b>Performance &amp; Diagnosis:</b> As anticipated in the assignment prompt, vanilla REINFORCE fails to converge on "
        "both continuous tasks. On <i>MountainCarContinuous-v0</i> (200,764 steps, lr=5e-5, γ=0.99), i.i.d. per-step Gaussian "
        "exploration almost never accumulates coherent momentum up the hill (+69.37 peak in a single episode at ~182k steps); "
        "because every step incurs -0.1 a², gradient updates drive actions toward zero (last-10 train -31.49 ± 1.28, "
        "deterministic eval -1.80 ± 0.01). On <i>BipedalWalker-v3</i> (201,432 steps), Monte-Carlo credit assignment across "
        "hundreds of joint torques produces noisy losses (±150) and high variance (-85.86 ± 30.36 train, -118.11 ± 18.92 eval).",
        body_style,
    ))
    story.append(KeepTogether([
        fit_image("part1_reinforce_losses.png", max_width=528, max_height=215),
        Paragraph("Figure 1: Part 1A REINFORCE policy loss across MountainCarContinuous-v0 (left) and BipedalWalker-v3 (right).", caption_style),
    ]))
    story.append(KeepTogether([
        fit_image("part1_mountaincar_reward.png", max_width=515, max_height=230),
        Paragraph("Figure 2: Part 1 learning curves on MountainCarContinuous-v0 (EMA smoothing 0.82; Gymnasium threshold +90.0).", caption_style),
    ]))

    # Part 1B
    story.append(PageBreak())

    # Part 1 Collective Comparative Analysis
    story.append(Paragraph("Collective Analysis for Part 1: REINFORCE vs. PPO vs. SAC", h1_style))
    story.append(Paragraph("3. Part 1B: Proximal Policy Optimization (PPO)", h1_style))
    story.append(Paragraph(
        "A rigorous side-by-side comparison reveals the fundamental theoretical trade-offs between on-policy Monte Carlo, on-policy actor-critic, and off-policy maximum entropy methods:",
        body_style
        "<b>Implementation:</b> <font name='DejaVuSansMono'>PPOAgent</font> uses decoupled 256-256 Actor and Critic MLPs "
        "with orthogonal initialization, Generalized Advantage Estimation (GAE), time-limit value bootstrapping, advantage "
        "standardization, gradient norm clipping (0.5), and clipped surrogate updates over 10 epochs per rollout.",
        body_style,
    ))

    table_data = [
        [Paragraph("<b>Dimension / Metric</b>", body_style), Paragraph("<b>REINFORCE (Part 1A)</b>", body_style), Paragraph("<b>PPO (Part 1B)</b>", body_style), Paragraph("<b>SAC (Part 1C Bonus)</b>", body_style)],
        [Paragraph("<b>Learning Paradigm</b>", body_style), Paragraph("On-policy Monte Carlo", body_style), Paragraph("On-policy Actor-Critic (GAE)", body_style), Paragraph("Off-policy Actor-Critic (Replay Buffer)", body_style)],
        [Paragraph("<b>Exploration Strategy</b>", body_style), Paragraph("State-independent Gaussian noise", body_style), Paragraph("Gaussian noise + entropy bonus", body_style), Paragraph("Maximum Entropy objective (&alpha; H(&pi;))", body_style)],
        [Paragraph("<b>Sample Efficiency</b>", body_style), Paragraph("Very Low (O(T) variance)", body_style), Paragraph("Moderate (Multi-epoch rollout buffer)", body_style), Paragraph("Very High (1 transition per grad step)", body_style)],
        [Paragraph("<b>Policy Constraint</b>", body_style), Paragraph("Unconstrained (High variance)", body_style), Paragraph("Clipped surrogate trust region (&epsilon;)", body_style), Paragraph("Soft policy iteration + twin Q min", body_style)],
        [Paragraph("<b>Target Networks</b>", body_style), Paragraph("None", body_style), Paragraph("None (GAE value baseline)", body_style), Paragraph("Twin target critics (Polyak &tau;=0.005)", body_style)],
        [Paragraph("<b>MountainCarContinuous-v0</b>", body_style), Paragraph("Fails (Trapped at valley floor ~-28.9)", body_style), Paragraph("Solves with tuning (~-16.3 to 0)", body_style), Paragraph("<b>Solves rapidly (~ -0.74, near-optimal)</b>", body_style)],
        [Paragraph("<b>BipedalWalker-v3</b>", body_style), Paragraph("Collapses (~ -67.7 to -110)", body_style), Paragraph("Learns stable gait (+109.5 tuned)", body_style), Paragraph("Rapid gait emergence (-16.4 early steps)", body_style)],
    ]

    t = Table(table_data, colWidths=[105, 135, 145, 155])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002D62')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D0D0D0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "<b>Part 1 Benchmark Empirical Curves:</b> The figures below showcase comparative episode return trajectories across both benchmark environments on the RCAC Scholar cluster:",
        body_style
        "<b>MountainCarContinuous-v0 (Solved, +95.17 ± 0.17):</b> With lr=3e-4, γ=0.999, λ=0.9, bs=64, buffered "
        "<font name='DejaVuSansMono'>RunningObsNorm</font>, and gSDE (<font name='DejaVuSansMono'>--use_sde</font>), PPO "
        "crossed +90.0 before 40k steps (+92.66 eval at 40,382 steps) and converged to +95.07 ± 0.28 train / +95.17 ± 0.17 "
        "held-out eval (PASS). Critic MSE fell cleanly from ~2,100 to &lt;0.1.<br/>"
        "<b>BipedalWalker-v3 Baseline (-109.56 ± 0.13):</b> With conservative starter defaults (lr=5e-5, bs=256, 200,424 steps), "
        "PPO improved slowly from -110 to +39.97 ± 17.30 in training (max +76.27), establishing the Part 2 tuning baseline.",
        body_style,
    ))
    story.append(KeepTogether([
        fit_image("part1_ppo_losses.png", max_width=528, max_height=290),
        Paragraph("Figure 3: Part 1B PPO Actor and Critic MSE losses on MountainCarContinuous-v0 (top) and BipedalWalker-v3 (bottom).", caption_style),
    ]))
    story.append(KeepTogether([
        fit_image("part1_bipedal_reward.png", max_width=515, max_height=225),
        Paragraph("Figure 4: Part 1 learning curves on BipedalWalker-v3 comparing REINFORCE, PPO Baseline, and SAC Bonus (+300.0 threshold).", caption_style),
    ]))

    # Embed Part 1 Reward Plots
    mc_plot = os.path.join(PLOTS_DIR, "part1_mountaincar_reward.png")
    bw_plot = os.path.join(PLOTS_DIR, "part1_bipedal_reward.png")
    if os.path.exists(mc_plot):
        story.append(Image(mc_plot, width=470, height=220))
        story.append(Spacer(1, 6))
    if os.path.exists(bw_plot):
        story.append(Image(bw_plot, width=470, height=220))
        story.append(Spacer(1, 6))

    # Part 1C Bonus
    story.append(PageBreak())

    # Dedicated SAC Bonus Section with Loss and Reward Diagnostics
    story.append(Paragraph("Part 1C (BONUS) Detailed Empirical Verification: SAC Diagnostics", h1_style))
    story.append(Paragraph("4. Part 1C (BONUS): Soft Actor-Critic (SAC) & Collective Part 1 Analysis", h1_style))
    story.append(Paragraph(
        "As required by the assignment specification for Part 1C (Soft Actor-Critic [2 pts]), we report both <b>reward</b> and <b>loss curves</b> "
        "(Critic MSBE loss, Actor policy loss, and temperature &alpha;) extracted directly from TensorBoard event logs on the RCAC Scholar cluster.",
        body_style
        "<b>Implementation:</b> <font name='DejaVuSansMono'>SACAgent</font> implements reparameterized squashed Gaussian "
        "actions a = tanh(u) with exact tanh-Jacobian log-probability correction, twin clipped Q-networks (Q<sub>1</sub>, Q<sub>2</sub>) "
        "with Polyak target updates (τ=0.01), detached learnable temperature α tuned toward target entropy -dim(A), and a "
        "NumPy ring replay buffer storing true termination <font name='DejaVuSansMono'>float(terminated)</font>.",
        body_style,
    ))

    story.append(Paragraph("1. SAC on MountainCarContinuous-v0 (Reward & Losses)", h2_style))
    story.append(Paragraph(
        "MountainCarContinuous presents a notorious sparse reward hurdle: the car must build momentum by swinging back and forth. "
        "Standard policy gradients fail because the car receives a continuous fuel penalty and cannot discover the goal summit. "
        "SAC's maximum entropy objective encourages high exploratory velocity swings early in training. As shown below, SAC converges to a total reward of <b>-0.74</b>, "
        "reflecting an almost optimal ascent trajectory with minimal fuel consumption. The critic loss rapidly drops and stabilizes near zero as the twin Q networks accurately model the value surface.",
        body_style
        "<b>MountainCarContinuous-v0 (Solved, +94.98 ± 0.33):</b> With lr=3e-4, τ=0.01, α<sub>init</sub>=0.5, start_steps=5,000, "
        "and OU warmup noise, SAC solved the hill climb immediately after warmup (+94.94 eval at 40k steps, +94.98 ± 0.33 held-out eval) "
        "while α decayed smoothly from 0.50 to ~0.07.<br/>"
        "<b>BipedalWalker-v3 (Solved, +328.79 ± 0.70):</b> With lr=7e-4, τ=0.01, bs=256, buffer=300k, and start_steps=10,000 "
        "over 500,509 steps, SAC crossed +300 at step 152,071 (+303.26 eval) and reached +328.86 final TB eval "
        "(<b>+328.79 ± 0.70</b> over 20/20 held-out seeds; <b>+326.64 ± 21.98</b> with 99/100 ≥ 300 over 100 seeds — PASS).",
        body_style,
    ))
    story.append(KeepTogether([
        fit_image("bonus_sac_mountaincar_diagnostics.png", max_width=528, max_height=175),
        Paragraph("Figure 5: SAC return, eval trajectory, MSBE critic loss, policy loss, and α on MountainCarContinuous-v0.", caption_style),
    ]))
    story.append(KeepTogether([
        fit_image("bonus_sac_bipedal_diagnostics.png", max_width=528, max_height=175),
        Paragraph("Figure 6: SAC return, eval trajectory, MSBE critic loss, policy loss, and α on BipedalWalker-v3.", caption_style),
    ]))

    sac_mc_diag = os.path.join(PLOTS_DIR, "bonus_sac_mountaincar_diagnostics.png")
    if os.path.exists(sac_mc_diag):
        story.append(Image(sac_mc_diag, width=520, height=200))
        story.append(Spacer(1, 6))
    story.append(Paragraph("Collective Analysis for Part 1 (REINFORCE vs. PPO vs. SAC)", h2_style))
    story.append(Paragraph("• <b>Sample Efficiency:</b> SAC &gt;&gt; PPO &gt;&gt; REINFORCE. Off-policy replay reuse lets SAC exceed +300 on BipedalWalker-v3 in ~152k steps, whereas on-policy PPO requires ~800k–2.5M steps (after tuning) and REINFORCE fails within 200k steps.", bullet_style))
    story.append(Paragraph("• <b>Gradient Variance &amp; Credit Assignment:</b> REINFORCE updates on unbaselined episode returns, failing on 1,600-step trajectories. PPO's critic + GAE cuts variance sharply, while SAC's twin clipped TD targets remove full-horizon variance altogether.", bullet_style))
    story.append(Paragraph("• <b>Exploration:</b> Standard i.i.d. Gaussian noise fails sparse/costly momentum tasks like MountainCarContinuous-v0; temporally correlated exploration (gSDE in PPO, OU warmup + entropy regularization in SAC) solves the task within 20k–40k steps.", bullet_style))

    story.append(Paragraph("2. SAC on BipedalWalker-v3 (Reward & Losses)", h2_style))
    # Part 2A
    story.append(PageBreak())
    story.append(Paragraph("5. Part 2A: PPO Hyperparameter Sweep on BipedalWalker-v3", h1_style))
    story.append(Paragraph(
        "BipedalWalker-v3 contains 24 continuous state observations and 4 continuous motor torques. "
        "SAC leverages its off-policy replay buffer (100,000 transitions) and batch size of 256 to conduct gradient descent on every environment step. "
        "The dual temperature &alpha; dynamically adapts to maintain target entropy H&#772; = -4.0. "
        "The critic loss shows stable convergence without divergence, enabling the agent to learn coordinated knee and hip motor actuation, achieving episode returns of <b>-16.42</b> within early step budgets.",
        body_style
        "To systematically diagnose PPO's Part 1B bottleneck on <i>BipedalWalker-v3</i>, four hyperparameters were swept "
        "individually for 200k environment steps (seed=42, rollout_steps=2048): <b>clip ratio</b> ∈ {0.1, 0.2, 0.3}, "
        "<b>entropy coefficient</b> ∈ {0.0, 0.01, 0.05}, <b>learning rate</b> ∈ {1e-5, 5e-5, 3e-4, 1e-3}, and "
        "<b>mini-batch size</b> ∈ {64, 128, 256} (evaluated at lr=3e-4 where updates are non-trivial).",
        body_style,
    ))

    sac_bw_diag = os.path.join(PLOTS_DIR, "bonus_sac_bipedal_diagnostics.png")
    if os.path.exists(sac_bw_diag):
        story.append(Image(sac_bw_diag, width=520, height=200))
        story.append(Spacer(1, 6))
    p2a_headers = ["Ablation", "Fixed Hyperparameters", "Value", "Steps", "Train Last-10", "Max Train", "Final Eval", "Max Eval"]
    p2a_rows = [
        ["Clip Ratio", "lr=5e-5, ent=0.01, bs=256", "0.1", "200,366", "-57.77 ± 29.16", "-35.12", "-27.13", "-27.13"],
        ["Clip Ratio", "lr=5e-5, ent=0.01, bs=256", "<b>0.2</b>", "200,424", "<b>+39.97 ± 17.30</b>", "<b>+76.27</b>", "-109.55", "-6.90"],
        ["Clip Ratio", "lr=5e-5, ent=0.01, bs=256", "0.3", "201,223", "+18.77 ± 45.83", "+48.89", "-115.42", "+71.14"],
        ["Entropy Coef", "lr=5e-5, clip=0.2, bs=256", "0.0", "200,891", "-72.43 ± 39.75", "-26.60", "-34.05", "-20.66"],
        ["Entropy Coef", "lr=5e-5, clip=0.2, bs=256", "<b>0.01</b>", "200,424", "<b>+39.97 ± 17.30</b>", "<b>+76.27</b>", "-109.55", "-6.90"],
        ["Entropy Coef", "lr=5e-5, clip=0.2, bs=256", "0.05", "201,148", "-35.11 ± 60.20", "+21.41", "+4.74", "+4.74"],
        ["Learning Rate", "clip=0.2, ent=0.01, bs=256", "1e-5", "201,122", "-91.51 ± 20.59", "-51.83", "-102.60", "-91.96"],
        ["Learning Rate", "clip=0.2, ent=0.01, bs=256", "5e-5", "200,424", "+39.97 ± 17.30", "+76.27", "-109.55", "-6.90"],
        ["Learning Rate", "clip=0.2, ent=0.01, bs=256", "3e-4", "200,466", "+44.79 ± 113.55", "+169.90", "+42.72", "+161.31"],
        ["Learning Rate", "clip=0.2, ent=0.01, bs=256", "<b>1e-3</b>", "200,120", "<b>+197.84 ± 61.07</b>", "<b>+235.33</b>", "<b>+279.44</b>", "<b>+279.44</b>"],
        ["Batch Size", "lr=3e-4, clip=0.2, ent=0.01", "64", "200,642", "+42.97 ± 98.19", "+120.67", "+39.62", "+128.11"],
        ["Batch Size", "lr=3e-4, clip=0.2, ent=0.01", "128", "201,155", "+45.29 ± 142.98", "<b>+205.80</b>", "<b>+81.44</b>", "+148.29"],
        ["Batch Size", "lr=3e-4, clip=0.2, ent=0.01", "<b>256</b>", "200,466", "+44.79 ± 113.55", "+169.90", "+42.72", "<b>+161.31</b>"],
    ]
    story.append(make_table(p2a_headers, p2a_rows, [62, 114, 34, 46, 82, 50, 50, 50]))
    story.append(Spacer(1, 6))

    story.append(KeepTogether([
        fit_image("part2a_clip_ratio_ablation.png", max_width=475, max_height=200),
        Paragraph("Figure 7: Part 2A Clip Ratio ablation (0.2 avoids under-updating at 0.1 and high-variance jumps at 0.3).", caption_style),
    ]))
    story.append(PageBreak())
    story.append(KeepTogether([
        fit_image("part2a_entropy_coef_ablation.png", max_width=475, max_height=195),
        Paragraph("Figure 8: Part 2A Entropy Coefficient ablation (0.01 improves early 200k discovery over 0.0 and 0.05).", caption_style),
    ]))
    story.append(KeepTogether([
        fit_image("part2a_lr_ablation.png", max_width=475, max_height=195),
        Paragraph("Figure 9: Part 2A Learning Rate ablation (lr=1e-3 reaches +279.44 eval within 200k steps vs -109.55 baseline).", caption_style),
    ]))
    story.append(KeepTogether([
        fit_image("part2a_batch_size_ablation.png", max_width=475, max_height=195),
        Paragraph("Figure 10: Part 2A Mini-Batch Size ablation at lr=3e-4 (bs=256 yields higher peak eval +161.31 and smoother updates).", caption_style),
    ]))

    # Part 2: Hyperparameter Analysis
    story.append(Paragraph("Part 2: Agent and Hyperparameter Analysis", h1_style))
    story.append(Paragraph("<b>Selected Environment:</b> BipedalWalker-v3 &nbsp;|&nbsp; <b>Selected Agent:</b> PPO", subtitle_style))

    story.append(Paragraph("Part 2A: Hyperparameter Tuning Analysis", h2_style))
    # Part 2B
    story.append(PageBreak())
    story.append(Paragraph("6. Part 2B: Best Tuned PPO vs. Part 1 Baseline", h1_style))
    story.append(Paragraph(
        "To systematically evaluate optimization dynamics in robotic locomotion, an ablation sweep was conducted across three hyperparameter axes (3 test values each):",
        body_style
        "<b>Tuned Configuration:</b> Combining Part 2A's strongest findings (<font name='DejaVuSansMono'>lr=1e-3, clip=0.2, bs=256</font>) "
        "with three stability and horizon improvements solved <i>BipedalWalker-v3</i> cleanly:<br/>"
        "• <b><font name='DejaVuSansMono'>rollout_steps = 8192</font>:</b> Single-env 2,048-step rollouts contain only ~1.3 trajectories, "
        "causing single terrain stumbles to bias updates or force slow 1,590-step gaits (~298 plateau). Collecting 8,192 steps (~6–7 full "
        "episodes) per PPO update averages gradients across multiple terrain seeds.<br/>"
        "• <b><font name='DejaVuSansMono'>gamma = 0.995, gae_lambda = 0.95</font>:</b> Extends planning horizon from 100 to 200 steps so fast "
        "finish-line strides outperform slow shuffling.<br/>"
        "• <b><font name='DejaVuSansMono'>entropy_coef = 0.0</font> + linear <font name='DejaVuSansMono'>anneal_lr</font> (2.5M steps, "
        "grad_clip=0.5):</b> Enables fast initial learning at 1e-3 followed by smooth annealing into a low-variance gait.",
        body_style,
    ))
    story.append(Paragraph(
        "1. <b>Clip Ratio (&epsilon; &in; {0.1, 0.2, 0.3}):</b><br/>"
        "• &epsilon; = 0.1 (Overly Conservative): Over-constrains policy divergence. While preventing catastrophic falls, it severely dampens gait progression (mean reward ~ -91.44).<br/>"
        "• &epsilon; = 0.2 (Optimal Trade-off): Balances policy improvement against distributional drift, enabling smooth cyclic locomotion (baseline ~ -57.51).<br/>"
        "• &epsilon; = 0.3 (Excessively Aggressive): Permissive policy jumps induce occasional uncontrolled knee overextensions, leading to stumbling episodes (~ -67.77).",
        bullet_style
        "<b>Final Performance:</b> Tuned PPO crossed +300 by step 802,524 (+305.08 eval), peaked at +308.01 eval (2.41M steps), "
        "and finished at 2,500,034 steps with Last-10 training return <b>+306.50 ± 1.79</b> (max +311.92) and TB final eval "
        "<b>+306.66</b> — a +417-point evaluation jump over the Part 1B baseline (-109.56). On held-out evaluation, it achieved "
        "<b>+307.75 ± 1.48</b> across 20/20 episodes (min 305.62, max 310.34) and <b>+301.54 ± 44.63</b> (98/100 ≥ 300) across 100 episodes (<b>PASS</b>).",
        body_style,
    ))
    story.append(Paragraph(
        "2. <b>Entropy Regularization Coefficient (c<sub>2</sub> &in; {0.0, 0.01, 0.05}):</b><br/>"
        "• c<sub>2</sub> = 0.0 (No Entropy): The policy collapses immediately into a deterministic local minimum—falling forward instantly to terminate the episode and avoid accumulating running torque penalties (reward ~ -108.58).<br/>"
        "• c<sub>2</sub> = 0.01 (Optimal Regularization): Sustains exploratory perturbation across all 4 leg motors while allowing the actor mean &mu;<sub>&theta;</sub> to converge to a fluid stride.<br/>"
        "• c<sub>2</sub> = 0.05 (Excessive Noise): Actuators jitter violently, breaking ground friction contact and depressing forward travel velocity (~ -82.53).",
        bullet_style
    ))
    story.append(Paragraph(
        "3. <b>Learning Rate (&alpha; &in; {1e-5, 5e-5, 3e-4}):</b><br/>"
        "• &alpha; = 1e-5 (Sluggish): Parameter updates are too small to resolve value estimation error in the critic (~ -70.88).<br/>"
        "• &alpha; = 5e-5 (Conservative Baseline): Stable and steady, but requires substantial environment rollouts (~ -57.51).<br/>"
        "• &alpha; = 3e-4 (Optimal Deep RL Standard): Rapidly aligns actor and critic representations, accelerating locomotion emergence (~ -36.33 in early epochs, paving the way for Part 2B).",
        bullet_style
    ))
    story.append(KeepTogether([
        fit_image("part2b_best_vs_baseline.png", max_width=515, max_height=245),
        Paragraph("Figure 11: Part 2B Tuned PPO (2.5M steps, rollout_steps=8192, lr=1e-3→0) vs. Part 1B Baseline on BipedalWalker-v3.", caption_style),
    ]))

    # Embed Part 2A Plots
    for p_name in [os.path.join(PLOTS_DIR, "part2a_clip_ratio_ablation.png"),
                   os.path.join(PLOTS_DIR, "part2a_entropy_coef_ablation.png"),
                   os.path.join(PLOTS_DIR, "part2a_lr_ablation.png")]:
        if os.path.exists(p_name):
            story.append(Image(p_name, width=470, height=210))
            story.append(Spacer(1, 4))

    story.append(Paragraph("Part 2B: Tuned Hyperparameter List vs. Baseline", h2_style))
    story.append(Paragraph(
        "<b>Optimal Hyperparameter Configuration:</b> Learning Rate = <code>3e-4</code>, Clip Ratio = <code>0.2</code>, "
        "Entropy Coefficient = <code>0.01</code>, Mini-batch Size = <code>128</code>, Rollout Steps = <code>2048</code>, GAE &lambda; = <code>0.95</code>, Discount &gamma; = <code>0.99</code>.<br/>"
        "<b>Empirical Superiority:</b> By reducing mini-batch size to 128 (increasing stochastic gradient updates per rollout from 80 to 160) and raising the learning rate to 3e-4, "
        "the tuned PPO agent dramatically outperforms the baseline, breaking the zero-reward threshold and achieving <b>+109.46</b> (solving stable continuous bipedal walking across uneven terrain).",
        body_style
    ))

    best_plot = os.path.join(PLOTS_DIR, "part2b_best_vs_baseline.png")
    if os.path.exists(best_plot):
        story.append(Image(best_plot, width=470, height=220))
        story.append(Spacer(1, 6))

    # Submission Checklist
    story.append(Paragraph("Submission Artifacts & Cluster Verification", h1_style))
    story.append(Paragraph(
        "• <b>Python Source Code:</b> Fully implemented, documented, and tested (<code>reinforce.py</code>, <code>ppo.py</code>, <code>sac.py</code>, <code>main.py</code>, <code>utils.py</code>, <code>logger.py</code>).<br/>"
        "• <b>Cluster Execution:</b> Executed on Purdue RCAC Scholar cluster using Slurm job submission scripts (<code>submit_all_jobs.sh</code>, <code>run_part2_sweep.sh</code>).<br/>"
        "• <b>TensorBoard Logs:</b> Complete run logs retained under <code>runs/</code>. Baseline Part 1 and tuned Part 2B runs organized in <code>selected_runs/</code>.<br/>"
        "• <b>Submission Zip Archive:</b> Compiled cleanly as <code>tri_vo_hw2.zip</code> containing code, <code>selected_runs/</code>, and this PDF report.",
        body_style
    ))

    doc.build(story)
    print(f"Successfully generated publication-grade PDF: {PDF_FILENAME}")
    print(f"PDF successfully generated: {PDF_FILENAME}")


if __name__ == '__main__':
if __name__ == "__main__":
    build_pdf()
