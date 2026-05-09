# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

datas = [('/home/shuhang/YBJ/RL_test/web/model.onnx', 'models'), ('/home/shuhang/YBJ/RL_test/web/model_v1.onnx', 'models'), ('/home/shuhang/YBJ/RL_test/web/model_v1_selfplay.onnx', 'models'), ('/home/shuhang/YBJ/RL_test/web/model_rainbow.onnx', 'models'), ('/home/shuhang/YBJ/RL_test/web/model_ppo.onnx', 'models')]
binaries = []
hiddenimports = ['slimevolleygym', 'slimevolleygym.slimevolley', 'gym', 'gym.envs.registration', 'onnxruntime', 'pygame']
datas += collect_data_files('slimevolleygym')
datas += collect_data_files('onnxruntime')
datas += collect_data_files('pygame')
binaries += collect_dynamic_libs('onnxruntime')
binaries += collect_dynamic_libs('pygame')
hiddenimports += collect_submodules('slimevolleygym')


a = Analysis(
    ['/home/shuhang/YBJ/RL_test/scripts/play_local.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='slime_play',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
