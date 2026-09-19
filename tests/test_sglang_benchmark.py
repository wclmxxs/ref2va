import copy
from types import SimpleNamespace
import pytest
from scripts.benchmark_sglang_acceleration import benchmark, sample, summary


def test_benchmark_preserves_case_separates_profiles_and_resumes_without_posts(tmp_path):
    calls=[]
    health={'ready':True,'instance':'same','metrics_schema_version':12,'profile':{'softmax_ranks':5},
            'request_options':{'optimizations':{'linear_kv_keep_ratio':1.}}}
    submissions={}
    def http(path,body=None):
        if path=='/openvdn/health':return copy.deepcopy(health)
        if path.endswith('/query/video_generation'):
            request=submissions[body['task_id']]
            opt=request['optimization']
            slow=opt['profile']
            return {'task':{'id':body['task_id'],'status':'succeeded','optimizations':{'requested':opt},
                'timings':{'denoise_seconds':30 if slow else 6,'gpu_worker_seconds':32 if slow else 8,'worker_wall_seconds':33 if slow else 9},
                'compilation':{'runtime_graph_reused':True,'compiled_new_graph':False},
                'cache_dit':{'cached_steps':[]},'content':{'url':'http://test/video'},'profiling':{'by_rank':[]}}}
        calls.append(body);task_id=str(len(calls));submissions[task_id]=body
        return {'task_id':task_id}
    client=SimpleNamespace(http=http,timeout=1)
    body={'model':'MiniMax-H3','seed':42,'duration':10,'ratio':'9:16','resolution':'768P',
          'content':[{'type':'text','text':'keep this'}], 'optimization':{'cache_dit':{'enabled':False}}}
    original=copy.deepcopy(body)
    result=benchmark(client,body,tmp_path,['native','combined','ulysses_dual'],repeat=2,kernels=True)
    assert body==original
    assert all(row['denoise_seconds']==6 and row['runs']==2 for row in result)
    assert all(request['content']==body['content'] and request['seed']==42 for request in calls)
    assert all(request['optimization']['cache_dit']=={'enabled':False} for request in calls)
    count=len(calls)
    assert benchmark(client,body,tmp_path,['native','combined','ulysses_dual'],repeat=2,kernels=True)==result
    assert len(calls)==count
    health['instance']='new'
    with pytest.raises(RuntimeError,match='Worker changed'):benchmark(client,body,tmp_path,['native'])


def test_uncertain_submission_never_duplicates(tmp_path):
    (tmp_path/'request.json').write_text('{}')
    client=SimpleNamespace(http=lambda path:{'ready':True,'instance':'same'})
    with pytest.raises(RuntimeError,match='Uncertain POST'):sample(client,{},tmp_path,'same')


def test_compiling_hot_runs_do_not_produce_speed_claims():
    records=[dict(variant='native',kind='hot',graph_reused=False,compiled_new_graph=True,
                  denoise_seconds=100,gpu_worker_seconds=102,worker_wall_seconds=105,cached_steps=[],video_url='x')]
    assert summary(records)[0]['denoise_seconds'] is None
