const fs = require('fs');
const crypto = require('crypto');
const source = 'Z:/orchestrator/services/a1111PromptMode.js';
const {applyA1111PromptMode} = require(source);
const graph = {
  '1': {class_type:'CheckpointLoaderSimple', inputs:{ckpt_name:'test.safetensors'}},
  '2': {class_type:'CLIPTextEncode', inputs:{clip:['1',1],text:'(watercolor:1.3), [night:day:0.5] AND portrait BREAK [soft]'}},
  '3': {class_type:'CLIPTextEncode', inputs:{clip:['1',1],text:'blur AND grain'}},
  '4': {class_type:'KSamplerAdvanced', inputs:{model:['1',0],positive:['2',0],negative:['3',0],steps:27}},
  '5': {class_type:'CLIPTextEncode', inputs:{clip:['1',1],text:'[a|b]'}},
  '6': {class_type:'KSamplerAdvanced', inputs:{model:['1',0],positive:['5',0],negative:['3',0],steps:12}},
};
const output = {source:'services/a1111PromptMode.js',sha256:crypto.createHash('sha256').update(fs.readFileSync(source)).digest('hex'),graph};
for (const noise of [false,true]) {
  const result = JSON.parse(JSON.stringify(graph));
  applyA1111PromptMode(result,{promptMode:'a1111',a1111NoiseSeed:noise,a1111Ensd:31337},{nodesAvailable:true});
  output[noise?'gpu':'cpu'] = result;
}
fs.writeFileSync('tests/data/a1111_site.json',JSON.stringify(output,null,2)+'\n');
