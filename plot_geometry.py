import numpy as np
import matplotlib.pyplot as plt
from config import Config
from geometry import inner_wall_radii, outer_wall_radii

cfg = Config()

z = np.linspace(0, cfg.L_domain, 500)
r_inner_low = [inner_wall_radii(zi, cfg)[0] * 1e3 for zi in z]
r_inner_high = [inner_wall_radii(zi, cfg)[1] * 1e3 for zi in z]
r_outer_low = [outer_wall_radii(zi, cfg)[0] * 1e3 for zi in z]
r_outer_high = [outer_wall_radii(zi, cfg)[1] * 1e3 for zi in z]

fig, ax = plt.subplots(figsize=(10, 6))

# Зеркалирование для полного сечения
ax.fill_between(z * 1e3, r_inner_low, r_inner_high, color='gray', alpha=0.7, label='Inner wall')
ax.fill_between(z * 1e3, -np.array(r_inner_high), -np.array(r_inner_low), color='gray', alpha=0.7)
ax.fill_between(z * 1e3, r_outer_low, r_outer_high, color='black', alpha=0.9, label='Outer wall')
ax.fill_between(z * 1e3, -np.array(r_outer_high), -np.array(r_outer_low), color='black', alpha=0.9)

ax.set_xlabel('z, мм')
ax.set_ylabel('r, мм')
ax.set_title(f'Геометрия капилляра\nL_transition={cfg.L_transition*1e3:.2f} мм, sharpness={cfg.sharpness:.1f}')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/geometry.png', dpi=200)
plt.show()
