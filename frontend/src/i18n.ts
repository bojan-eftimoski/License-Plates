// Lightweight EN/MK strings for the UI. The classical pipeline, endpoints and data flow are
// unchanged — this only localises visible text. {placeholders} are filled by makeT(...).

export type Lang = 'en' | 'mk'
type Dict = Record<string, string>

const en: Dict = {
  app_title: '🚗 Macedonian Plate Reader',
  app_sub: 'Classical computer vision · detect & read MK license plates',
  tab_upload: 'Upload',
  tab_camera: 'Live camera',
  backend_url: 'Backend URL',
  status_ready: 'backend ready',
  status_connecting: 'waking backend…',
  status_offline: 'backend offline',
  footer: 'Detect-always, read-when-confident · graded model runs on the backend (cv2 + numpy, no deep learning)',

  drop_prompt: 'Choose, drag & drop, or paste an image/video…',
  analyzing: 'Analyzing…',
  analyze: 'Analyze',
  drop_invalid: 'Drop an image or video file.',
  backend_unreachable: "Couldn't reach the backend at {base}. Is it running / awake? ({err})",
  plates_read: 'Plates read: {n}',
  frames_scanned: ' · {frames} frames scanned',
  th_plate: 'Plate',
  th_region: 'Region',
  th_confidence: 'Confidence',
  no_read: "No plate read confidently. Red boxes mark detected plate regions the model couldn't read (too small / blurred / angled); green boxes are confident reads.",
  copied: '✓ Copied',
  copy_csv: '⧉ Copy CSV',
  download_csv: '⬇ Download results CSV',
  clipboard_blocked: 'Clipboard blocked by the browser — use the download link instead.',

  cam_insecure: 'Camera needs a secure (HTTPS) page. Open the github.io URL, not an http:// address or IP.',
  cam_unsupported: 'This browser blocks camera access here. Use Safari (iOS) or Chrome/Edge (desktop) on the HTTPS site.',
  cam_busy: 'Camera is busy or blocked. Close other apps/tabs using it (Zoom, Teams, Windows Camera, Photo Booth), then check your OS camera privacy setting and retry. On Windows: Settings → Privacy & security → Camera → on.',
  cam_denied: 'Camera permission was denied. iPhone: tap "aA" in Safari\'s address bar → Website Settings → Camera → Allow, then reload. Desktop: click the camera icon in the address bar → Allow.',
  cam_none: 'No camera was found on this device. Connect one (or use the Upload tab) and retry.',
  cam_generic: 'Camera error: {n}. Needs HTTPS + an available camera.',
  cam_opening: 'Opening camera…',
  cam_stop: 'Stop',
  cam_start: 'Start camera',
  cam_fullscreen: '⛶ Fullscreen',
  cam_exit: 'Exit fullscreen',
  cam_hint: 'Point the rear camera at a Macedonian plate. Detection runs on the backend ~2×/s; the label glides and locks via multi-frame voting, and lingers ~3s after the plate leaves frame. Works best on clear, close plates.',
}

const mk: Dict = {
  app_title: '🚗 Читач на регистарски таблички',
  app_sub: 'Класична компјутерска визија · детекција и читање на македонски таблички',
  tab_upload: 'Прикачи',
  tab_camera: 'Камера во живо',
  backend_url: 'Адреса на бекенд',
  status_ready: 'бекендот е спремен',
  status_connecting: 'будење на бекендот…',
  status_offline: 'бекендот е офлајн',
  footer: 'Секогаш детектира, чита кога е сигурно · оценуваниот модел работи на бекендот (cv2 + numpy, без длабоко учење)',

  drop_prompt: 'Избери, влечи и пушти, или залепи слика/видео…',
  analyzing: 'Анализирам…',
  analyze: 'Анализирај',
  drop_invalid: 'Пушти слика или видео датотека.',
  backend_unreachable: 'Не може да се поврзе со бекендот на {base}. Дали работи / е разбуден? ({err})',
  plates_read: 'Прочитани таблички: {n}',
  frames_scanned: ' · скенирани {frames} рамки',
  th_plate: 'Табличка',
  th_region: 'Регион',
  th_confidence: 'Сигурност',
  no_read: 'Ниту една табличка не е прочитана со сигурност. Црвените правоаголници означуваат детектирани таблички што моделот не можеше да ги прочита (премали / заматени / под агол); зелените се сигурни читања.',
  copied: '✓ Копирано',
  copy_csv: '⧉ Копирај CSV',
  download_csv: '⬇ Преземи CSV резултати',
  clipboard_blocked: 'Прелистувачот ја блокира таблата со исечоци — користи го линкот за преземање.',

  cam_insecure: 'Камерата бара безбедна (HTTPS) страница. Отвори ја github.io адресата, не http:// адреса или IP.',
  cam_unsupported: 'Овој прелистувач го блокира пристапот до камерата. Користи Safari (iOS) или Chrome/Edge (десктоп) на HTTPS страницата.',
  cam_busy: 'Камерата е зафатена или блокирана. Затвори ги другите апликации/јазичиња што ја користат (Zoom, Teams, Windows Camera, Photo Booth), провери ја поставката за приватност на камерата и обиди се повторно. На Windows: Поставки → Приватност и безбедност → Камера → вклучено.',
  cam_denied: 'Пристапот до камерата е одбиен. iPhone: допри „aA“ во адресната лента на Safari → Website Settings → Camera → Allow, па освежи. Десктоп: кликни на иконата за камера во адресната лента → Allow.',
  cam_none: 'Не е пронајдена камера на овој уред. Поврзи камера (или користи го јазичето „Прикачи“) и обиди се повторно.',
  cam_generic: 'Грешка со камерата: {n}. Бара HTTPS + достапна камера.',
  cam_opening: 'Отворам камера…',
  cam_stop: 'Стоп',
  cam_start: 'Вклучи камера',
  cam_fullscreen: '⛶ Цел екран',
  cam_exit: 'Излез од цел екран',
  cam_hint: 'Насочи ја задната камера кон македонска табличка. Детекцијата работи на бекендот ~2×/сек; ознаката се движи и се заклучува преку гласање од повеќе рамки, и останува ~3 сек откако табличката ќе излезе од кадар. Најдобро работи на јасни, блиски таблички.',
}

export const STRINGS: Record<Lang, Dict> = { en, mk }

export function makeT(lang: Lang) {
  const d = STRINGS[lang]
  return (key: string, params?: Record<string, string | number>): string => {
    let s = d[key] ?? en[key] ?? key
    if (params) for (const k in params) s = s.replace(`{${k}}`, String(params[k]))
    return s
  }
}

export type T = ReturnType<typeof makeT>
