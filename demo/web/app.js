const $ = (id) => document.getElementById(id);
let current = null;
let uploadData = null;
const labels = {noisy:'Ảnh nhiễu', gaussian:'Gaussian blur', bilateral:'Bilateral', nlm:'Non-Local Means', fusion:'Learned fusion', mlp:'Residual MLP'};
const order = ['noisy','gaussian','bilateral','nlm','fusion','mlp'];
const format = (v,n=2) => Number(v).toFixed(n);

function setSplit() {
  const value = Number($('split').value);
  $('overlay-clip').style.clipPath = `inset(0 0 0 ${value}%)`;
  $('split-line').style.left = `${value}%`;
}
function setMethod() {
  if (!current) return;
  $('overlay-image').src = current.images[$('method').value];
  $('overlay-image').alt = `${labels[$('method').value]} cho ảnh ${current.id}`;
}
function renderMetrics(data) {
  const nlm = data.scores.nlm.psnr_db;
  const best = Math.max(...order.map(key => data.scores[key].psnr_db));
  $('metric-rows').innerHTML = order.map(key => {
    const score = data.scores[key];
    const delta = score.psnr_db - nlm;
    const deltaText = key === 'nlm' ? '—' : `${delta >= 0 ? '+' : ''}${format(delta)}`;
    const deltaClass = delta > .005 ? 'positive' : delta < -.005 ? 'negative' : '';
    return `<tr><td>${labels[key]}</td><td class="${score.psnr_db === best ? 'is-best' : ''}">${format(score.psnr_db)}</td><td>${format(score.ssim,4)}</td><td class="${deltaClass}">${deltaText}</td></tr>`;
  }).join('');
}
function renderBenchmark(summary, paired) {
  const methods = ['gaussian','bilateral','nlm','fusion','mlp'];
  $('benchmark-grid').innerHTML = [15,25,50].map(sigma => {
    const level = summary.filter(row => Number(row.sigma) === sigma);
    const values = methods.map(method => level.find(row => row.method === method));
    const min = Math.min(...values.map(row => Number(row.mean_psnr_db))) - .5;
    const max = Math.max(...values.map(row => Number(row.mean_psnr_db))) + .5;
    const pair = paired.find(row => Number(row.sigma) === sigma);
    return `<div class="bench-level"><h3>σ ${sigma}</h3>${values.map(row => `<div class="bench-row ${row.method}"><span>${labels[row.method]}</span><span class="bench-track"><span class="bench-fill" style="width:${Math.max(4,100*(Number(row.mean_psnr_db)-min)/(max-min))}%"></span></span><strong>${format(row.mean_psnr_db)}</strong></div>`).join('')}<p class="bench-delta">Fusion − NLM: ${Number(pair.mean_delta_db)>=0?'+':''}${format(pair.mean_delta_db)} dB · thắng ${pair.win_count}/${pair.n_images} ảnh</p></div>`;
  }).join('');
  $('benchmark-note').textContent = 'Thanh biểu diễn PSNR (dB) trong từng mức nhiễu; trục được thu gọn để thấy chênh lệch. Số đầy đủ và khoảng tin cậy bootstrap nằm trong results/.';
}
async function run() {
  $('run').disabled = true;
  $('status').textContent = 'Đang xử lý ảnh trên CPU…';
  try {
    const sigma = Number(document.querySelector('input[name="sigma"]:checked').value);
    const body = {sigma, example_id:$('example').value};
    if (uploadData) body.image = uploadData;
    const response = await fetch('/api/process',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Không thể xử lý ảnh');
    current = data;
    $('base-image').src = data.images.reference;
    $('base-image').alt = `Ảnh tham chiếu ${data.id}`;
    $('image-meta').textContent = `${data.id} · ${data.size} px`;
    $('sigma-meta').textContent = `σ = ${sigma}`;
    $('empty-stage').hidden = true;
    document.querySelector('.image-stage').classList.add('ready');
    setMethod();setSplit();renderMetrics(data);
    $('status').textContent = `Hoàn tất · ${data.size} px · 6 đầu ra`;
  } catch(error) {$('status').textContent = `Lỗi: ${error.message}`;}
  finally {$('run').disabled = false;}
}
async function init() {
  try {
    const response = await fetch('/api/bootstrap');
    if (!response.ok) throw new Error('Chưa có kết quả thí nghiệm');
    const data = await response.json();
    $('example').innerHTML = data.examples.map(id => `<option value="${id}">Ảnh #${id}</option>`).join('');
    renderBenchmark(data.summary,data.paired);
    await run();
  } catch(error) {$('status').textContent = `Lỗi khởi tạo: ${error.message}`;}
}
$('run').addEventListener('click',run);
$('method').addEventListener('change',setMethod);
$('split').addEventListener('input',setSplit);
$('example').addEventListener('change',()=>{uploadData=null;$('upload').value='';$('upload-name').textContent='Đang dùng ảnh mẫu Berkeley.';});
$('upload').addEventListener('change',async(event)=>{const file=event.target.files[0];if(!file)return;if(file.size>4_000_000){$('status').textContent='Ảnh vượt quá 4 MB.';return;}uploadData=await new Promise(resolve=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.readAsDataURL(file);});$('upload-name').textContent=`Đang dùng ${file.name} làm ảnh tham chiếu.`;});
init();
