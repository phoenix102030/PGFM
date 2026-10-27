from pathlib import Path
from datetime import datetime, timezone
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

P = Path(__file__).resolve().parent
DATA = P.parents[1] / 'data'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=14)
    parser.add_argument('--vmax', type=float, default=10.)
    args = parser.parse_args()
    with np.load(DATA / 'op_net_maps.npz') as z:
        lat, lon = z['lat'], z['lon']
        times = z['time']
        if not 0 <= args.frame < len(times):
            raise ValueError(f'frame must be 0..{len(times)-1}')
        values = z['net_op_column_rate_tecu_per_hour'][args.frame]
        stamp = datetime.fromtimestamp(int(times[args.frame]), timezone.utc)

    plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.7),
                             subplot_kw={'projection': 'polar'})
    theta = np.deg2rad(np.r_[lon - 1.25, lon[-1] + 1.25])
    latitude_edges = np.arange(45., 90.01, 2.5)
    radius = np.tan(np.deg2rad((90 - latitude_edges) / 2))
    norm = Normalize(-args.vmax, args.vmax)

    for ax, north, title in zip(axes, [True, False], ['North Pole', 'South Pole']):
        ids = np.where(lat >= 45 if north else lat <= -45)[0]
        ids = ids[np.argsort(np.abs(lat[ids]))]
        ax.grid(False)
        mesh = ax.pcolormesh(theta, radius, values[ids], cmap='RdBu_r',
                             norm=norm, shading='flat', rasterized=True)
        ax.set_theta_zero_location('S' if north else 'N')
        ax.set_theta_direction(1 if north else -1)
        ax.set_ylim(0, radius[0])
        ax.set_xticks(np.deg2rad([0, 60, 120, 180, 240, 300]))
        ax.set_xticklabels(['0°', '60°E', '120°E', '180°', '120°W', '60°W'])
        rings = np.array([80, 60])
        ax.set_yticks(np.tan(np.deg2rad((90 - rings) / 2)))
        ax.set_yticklabels([f'{v}°'+('N' if north else 'S') for v in rings])
        ax.set_rlabel_position(30)
        ax.grid(color='.4', alpha=.35, linewidth=.5)
        ax.set_title(title, pad=24)

    fig.subplots_adjust(left=.055, right=.86, bottom=.16, top=.80, wspace=.27)
    cax = fig.add_axes([.90, .20, .022, .54])
    cb = fig.colorbar(mesh, cax=cax, extend='both')
    cb.set_label(r'Net O$^+$ column rate of change (TECU/h)')
    fig.text(.46, .04, stamp.strftime('%d %B %Y, %H:%M UTC'), ha='center')
    for ext in ['png', 'pdf']:
        fig.savefig(P / f'pole_frame{args.frame:03d}.{ext}', dpi=300,
                    facecolor='white')
    plt.close(fig)

if __name__ == '__main__':
    main()
