import ast
import json
from pathlib import Path
import types
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from openvdn_comfy.config import Settings
from openvdn_comfy.conditioning import load_conditioning
from openvdn_comfy.keyframes import normalize_anchors,prepare_keyframes,conditioning_mode
from openvdn_comfy.runner import conditioning_key


def test_cache_distinguishes_anchors_and_canvas_but_preserves_reference_identity(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(64,32)).save(image)
    ref=conditioning_key('prompt',[image],768)
    assert ref==conditioning_key('prompt',[image],768,['ref'],(1376,768))
    first=conditioning_key('prompt',[image],768,['first'],(1376,768))
    last=conditioning_key('prompt',[image],768,['last'],(1376,768))
    resized=conditioning_key('prompt',[image],768,['first'],(768,1376))
    assert len({ref,first,last,resized})==4
    assert first==conditioning_key('prompt',[image],512,['first'],(1376,768))
    with pytest.raises(ValueError):conditioning_key('prompt',[image],768,['first'])
    for anchors in (['first','first'],['last','first'],['ref','first']):
        with pytest.raises(ValueError):normalize_anchors(2,anchors)
    assert conditioning_mode(normalize_anchors(0))=='t2va'


def test_keyframe_canvas_stretches_first_but_cover_crops_last():
    pixels=np.zeros((32,96,3),dtype=np.uint8);pixels[:,:32,0]=255;pixels[:,32:64,1]=255;pixels[:,64:,2]=255
    image=Image.fromarray(pixels)
    first,last=prepare_keyframes([image,image],32,32)
    assert first.size==last.size==(32,32)
    assert first.getpixel((0,16))[0]>240 and first.getpixel((31,16))[2]>240
    assert np.all(np.array(last)==[0,255,0])


def test_conditioner_writes_real_anchor_metadata_and_canvas_latents(tmp_path,monkeypatch):
    # Exercise the actual resident method; substitute only the heavy encoder/VAE.
    source=Path(__file__).resolve().parents[1]/'scripts/resident_worker.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Conditioner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='encode')
    ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
    encoder=types.ModuleType('src.inference.encode_keyframes')
    seen=[]
    def presentation(processor,prompt,images):
        seen.append([im.size for im in images]);return [1,2],[1,0],{'image_grid_thw':torch.tensor([[1,2,3]]*len(images))}
    encoder.build_presentation=presentation
    encoder.qwen3vl_prompt_embeds=lambda *args:torch.zeros(2,4,dtype=torch.bfloat16)
    encoder.encode_vae_condition=lambda vae,tensor,*args:torch.zeros(1,24,1,tensor.shape[-2]//16,tensor.shape[-1]//16)
    encoder.normalize_references=lambda imgs,size:imgs
    encoder.PIXEL_MEAN=encoder.PIXEL_STD=(0,0,0);encoder.KEYFRAME_ENCODE_SEED=42
    promptmod=types.ModuleType('src.inference.encode_prompt');promptmod.encode=lambda *args:None
    for name,module in [('src',types.ModuleType('src')),('src.inference',types.ModuleType('src.inference')),
                        ('src.inference.encode_keyframes',encoder),('src.inference.encode_prompt',promptmod)]:
        monkeypatch.setitem(sys.modules,name,module)
    obj=types.SimpleNamespace(processor=None,encoder=None,input_device='cpu',device='cpu',vae=None)
    refs=[]
    for i,size in enumerate([(120,80),(80,120)]):
        path=tmp_path/f'{i}.png';Image.new('RGB',size).save(path);refs.append(str(path))
    plan=Settings(duration=5,ratio='9:16',resolution=256).render_plan();output=tmp_path/'condition.pt'
    ns['encode'](obj,'prompt',refs,output,768,['first','last'],plan)
    _,_,conditions,metadata=load_conditioning(output,'cpu')
    assert conditions[0]==('first','last')
    assert all(tuple(c.shape[-2:])==(plan.generation_height//16,plan.generation_width//16) for c in conditions[1])
    assert seen==[[(plan.generation_width,plan.generation_height)]*2]
    assert metadata['image_anchors']==['first','last'] and metadata['reference_short_edge'] is None
