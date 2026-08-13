# -*- coding: utf-8 -*-
"""Shared, colorblind-audited palette and font policy for all manuscript figures (palette D)."""
import matplotlib.pyplot as plt
OK = {
    '闊葉樹林型':   '#8A4E20',
    '針葉樹林型':   '#14602E',
    '針闊葉樹混淆': '#9BB13C',
    '竹林':         '#2E8B37',
    '竹闊混淆林':   '#D4A020',
    '待成林地':     '#C67B2E',
}
STROKE = {
    '闊葉樹林型':   '#6E3D18',
    '針葉樹林型':   '#0F4A24',
    '針闊葉樹混淆': '#71862A',
    '竹林':         '#1F6E29',
    '竹闊混淆林':   '#A47C10',
    '待成林地':     '#9A5E1E',
}
EN = {
    '闊葉樹林型':'Broadleaf', '針葉樹林型':'Conifer', '針闊葉樹混淆':'Conifer–broadleaf',
    '竹林':'Bamboo', '竹闊混淆林':'Bamboo–broadleaf', '待成林地':'Immature',
}
BAMBOO   = OK['竹林'];       BAMBOO_D  = STROKE['竹林']
BROAD    = OK['闊葉樹林型'];  BROAD_D   = STROKE['闊葉樹林型']
GOLD     = OK['竹闊混淆林'];  GOLD_D    = STROKE['竹闊混淆林']
RUST     = OK['待成林地']
INK      = '#111111'
GREY     = '#555555'
SCAR_B   = '#C21807'
SCAR_O   = '#1F5FBF'
def set_fonts(base=9.5):
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':base,
        'axes.titlesize':base+1.5, 'axes.labelsize':base+0.5,
        'xtick.labelsize':base-1, 'ytick.labelsize':base-1,
        'legend.fontsize':base-1, 'axes.linewidth':0.9, 'figure.dpi':400,
    })
F_SMALL = 8.0; F_ANNOT = 8.6; F_VALUE = 9.0
print('palette loaded')
