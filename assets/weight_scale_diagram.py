import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.5)
ax.axis('off')

bar_y = 2.2
bar_h = 1.0
segments = [
    (0.5, 3.5, "#5b8bd4", "0–29"),
    (4.0, 5.0, "#2e5090", "30–69"),
    (9.0, 4.5, "#1a3a5c", "70–100"),
]

for (x, w, color, label) in segments:
    rect = mpatches.FancyBboxPatch(
        (x, bar_y), w, bar_h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#0a0a0a", linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, bar_y + bar_h / 2, label,
            ha='center', va='center', fontsize=32, fontweight='bold',
            color='white')

descriptions = [
    (0.5 + 3.5 / 2, "Cited only if directly\nand specifically relevant"),
    (4.0 + 5.0 / 2, "Cited only if\nclearly relevant"),
    (9.0 + 4.5 / 2, "Cited if relevant\nor possibly relevant"),
]

for x, desc in descriptions:
    ax.text(x, bar_y - 0.7, desc,
            ha='center', va='center', fontsize=18, fontweight='normal',
            color='#1a1a1a')

ax.text(7.0, bar_y + bar_h + 0.5, 'Document Weight Priority Scale', ha='center', va='center',
        fontsize=34, fontweight='bold', color='#1a1a1a')

fig.patch.set_alpha(0)
plt.tight_layout()
plt.savefig('/home/beams0/MADELINE.MILLER/Desktop/llm-xpcs-eval/assets/weight_scale_diagram.png',
            dpi=200, bbox_inches='tight', transparent=True)
plt.close()
print("Saved weight_scale_diagram.png")
