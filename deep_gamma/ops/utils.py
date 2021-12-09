

def remove_frame(ax, sides=["top", "left", "right"]):
    """Remove the frame of a matplotlib plot"""
    for side in sides:
        ax_side = ax.spines[side]
        ax_side.set_visible(False)