import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(13.54, 8))
ax.set_xlim(0.7, 14.3)
ax.set_ylim(0.1, 7.4)
ax.axis('off')

x_center = 7.5

boxes = [
    (6.5, "User Has Question"),
    (4.5, "Contact Scientist"),
    (2.5, "Wait for Response"),
]

colors = ["#1a3a5c", "#2e5090", "#5b8bd4"]
box_w = 10.0
box_h = 1.2
x_center = 7.5

for (y, label), color in zip(boxes, colors):
    rect = mpatches.FancyBboxPatch(
        (x_center - box_w / 2, y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="#0a0a0a", linewidth=3
    )
    ax.add_patch(rect)
    ax.text(x_center, y, label,
            ha='center', va='center', fontsize=44, fontweight='bold',
            color='white')

for i in range(len(boxes) - 1):
    y_start = boxes[i][0] - box_h / 2
    y_end = boxes[i + 1][0] + box_h / 2
    ax.annotate('', xy=(x_center, y_end), xytext=(x_center, y_start),
                arrowprops=dict(arrowstyle='-|>,head_length=1.5,head_width=1.0',
                                color='#1a1a1a', lw=16, mutation_scale=20))

loop_x = x_center + box_w / 2 + 0.8
ax.annotate('',
    xy=(x_center + box_w / 2, boxes[0][0]),
    xytext=(x_center + box_w / 2, boxes[2][0]),
    arrowprops=dict(
        arrowstyle='-|>,head_length=1.5,head_width=1.0',
        color='#cc0000', lw=10, mutation_scale=18,
        connectionstyle='arc3,rad=-0.4'
    ))

ax.text(loop_x + 0.6, boxes[1][0], 'Repeat', ha='center', va='center',
        fontsize=36, fontweight='bold', color='#cc0000', rotation=90)

fig.patch.set_alpha(0)
plt.tight_layout()
plt.savefig('/home/beams0/MADELINE.MILLER/Desktop/llm-xpcs-eval/assets/motivation_diagram.png',
            dpi=200, bbox_inches='tight', transparent=True)
plt.close()
print("Saved motivation_diagram.png")
