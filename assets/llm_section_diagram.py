import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(14, 3.5))
ax.set_xlim(0.2, 14.8)
ax.set_ylim(0.2, 3.3)
ax.axis('off')

boxes = [
    (0.5, "3-Phase\nRetrieval"),
    (4.2, "Document Weight\nRanking"),
    (7.9, "LLM\nReranker"),
    (11.6, "Cited\nResponse"),
]

colors = ["#1a3a5c", "#2e5090", "#3a6abf", "#5b8bd4"]
box_w = 2.8
box_h = 2.0
y_center = 1.75

for (x, label), color in zip(boxes, colors):
    rect = mpatches.FancyBboxPatch(
        (x, y_center - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="#0a0a0a", linewidth=3
    )
    ax.add_patch(rect)
    ax.text(x + box_w / 2, y_center, label,
            ha='center', va='center', fontsize=28, fontweight='bold',
            color='white')

for i in range(len(boxes) - 1):
    x_start = boxes[i][0] + box_w
    x_end = boxes[i + 1][0]
    ax.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                arrowprops=dict(arrowstyle='-|>,head_length=1.5,head_width=1.0',
                                color='#1a1a1a', lw=10, mutation_scale=20))

fig.patch.set_alpha(0)
plt.tight_layout()
plt.savefig('/home/beams0/MADELINE.MILLER/Desktop/llm-xpcs-eval/assets/llm_section_diagram.png',
            dpi=200, bbox_inches='tight', transparent=True)
plt.close()
print("Saved llm_section_diagram.png")
