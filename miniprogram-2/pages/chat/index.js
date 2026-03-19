const { baseUrl } = require('../../config.js');

function formatReqError(err) {
  if (err && err.message) return String(err.message);
  return '网络异常';
}

function parseStudentIdentity(text) {
  const v = String(text || '').trim();
  // 支持示例：张睿宸 37 / 张睿宸,37 / 张睿宸#37
  const m = v.match(/^([\u4e00-\u9fa5·]{2,20})[\s,，#|/:-]+([A-Za-z0-9_-]{1,32})$/);
  if (!m) return null;
  return { student_name: m[1], student_uid: m[2] };
}

Page({
  /** 页面的初始数据 */
  data: {
    openid: '',
    student_uid: '',
    phase: 'init', // init | need_name | pending | chatting

    messages: [], // { from: 0/1, content, time }
    input: '',
    anchor: '', // 消息列表滚动到 id 与之相同的元素的位置
    keyboardHeight: 0, // 键盘当前高度(px)
  },

  /** 生命周期函数--监听页面加载 */
  onLoad(options) {
    this.bootstrap();
  },

  /** 生命周期函数--监听页面初次渲染完成 */
  onReady() {},

  /** 生命周期函数--监听页面显示 */
  onShow() {},

  /** 生命周期函数--监听页面隐藏 */
  onHide() {},

  async _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  },

  async _fetchLoginCode() {
    return new Promise((resolve, reject) => {
      wx.login({
        success: (r) => resolve((r && r.code) || ''),
        fail: reject,
      });
    });
  },

  async bootstrap() {
    // 0) 服务端是否配置了小程序密钥（与开发者工具 AppID 是否一致可对照）
    try {
      const st = await this._req('/api/mini-status', 'GET');
      console.info('[mini-status]', st);
      if (st && st.wx_mini_configured === false) {
        this._pushBot(
          `【服务端】未配置微信小程序密钥。当前接口返回 appid=${(st && st.appid) || '无'}。请在服务器项目根目录 .env 填写 WX_MINI_APPID、WX_MINI_SECRET 后重启后端（.env 须与运行中的进程在同一套部署里）。`
        );
      } else if (st && st.appid) {
        try {
          const acc = wx.getAccountInfoSync();
          const localId = (acc && acc.miniProgram && acc.miniProgram.appId) || '';
          if (localId && localId !== st.appid) {
            this._pushBot(
              `【AppID 不一致】本机小程序为 ${localId}，服务器配置为 ${st.appid}。请把服务器 .env 的 WX_MINI_APPID 改成与当前小程序一致，或改用正确 AppID 打开本项目。`
            );
          }
        } catch (e) {
          console.warn('getAccountInfoSync', e);
        }
      }
    } catch (e) {
      console.error('/api/mini-status 失败', e);
      this._pushBot(
        `【网络】无法访问 /api/mini-status，请检查 config.js 的 baseUrl、HTTPS 证书与小程序 request 合法域名。详情：${formatReqError(e)}`
      );
    }

    // 1) wx.login → code，最多换 3 次 code 请求 /api/wx-login（code 一次性，失败可重试）
    let openid = '';
    for (let attempt = 0; attempt < 3 && !openid; attempt += 1) {
      if (attempt > 0) {
        await this._sleep(400);
      }
      let code = '';
      try {
        code = await this._fetchLoginCode();
      } catch (e) {
        console.error('wx.login 失败', e);
        break;
      }
      if (!code) continue;
      try {
        const r = await this._req('/api/wx-login', 'POST', { code });
        openid = (r && r.openid) ? String(r.openid) : '';
      } catch (e) {
        console.error('/api/wx-login 失败', attempt, formatReqError(e));
      }
    }

    if (!openid) {
      openid = wx.getStorageSync('openid') || '';
    }
    if (openid) {
      wx.setStorageSync('openid', openid);
    }
    this.setData({ openid });

    // 2) binding status
    try {
      const st = await this._req(`/api/binding-status?openid=${encodeURIComponent(openid)}`, 'GET');
      const approved = (st && st.approved_students) ? st.approved_students : [];
      if (approved && approved.length) {
        this.setData({ student_uid: approved[0], phase: 'chatting' });
        this._pushBot('已识别到你绑定的孩子信息，可以开始聊天了。');
      } else {
        this.setData({ phase: 'need_name' });
        this._pushBot('请先发送“孩子姓名 + HYDRO ID”（例如：张睿宸 37），通过后才可聊天。');
      }
    } catch (e) {
      this.setData({ phase: 'need_name' });
      this._pushBot('请先发送“孩子姓名 + HYDRO ID”（例如：张睿宸 37），通过后才可聊天。');
    }

    if (!(this.data.openid && String(this.data.openid).trim())) {
      this._pushBot(
        '【重要】当前未获取到微信 openid，发姓名会失败。请核对：① 服务器 .env 中 WX_MINI_APPID / WX_MINI_SECRET 与开发者工具里的 AppID 一致；② 已重启后端；③ 小程序后台已配置 request 合法域名；④ config.js 的 baseUrl 正确。可在 vConsole 查看 /api/wx-login 报错。'
      );
    }
    wx.nextTick(this.scrollToBottom);
  },

  _pushUser(content) {
    const messages = this.data.messages || [];
    messages.push({ from: 0, content, time: Date.now() });
    this.setData({ messages });
  },

  _pushBot(content) {
    const messages = this.data.messages || [];
    messages.push({ from: 1, content, time: Date.now() });
    this.setData({ messages });
  },

  _req(url, method = 'GET', data = undefined) {
    const root = (baseUrl || '').replace(/\/+$/, '');
    return new Promise((resolve, reject) => {
      wx.request({
        url: root + url,
        method,
        data,
        header: { 'content-type': 'application/json' },
        success: (res) => {
          const sc = res.statusCode || 0;
          if (sc >= 200 && sc < 300) {
            resolve(res.data);
            return;
          }
          const body = res.data;
          let msg = `HTTP ${sc}`;
          if (body && typeof body === 'object') {
            if (body.detail) {
              const d = body.detail;
              if (Array.isArray(d)) {
                msg = d
                  .map((x) => (x && (x.msg || x.message)) || JSON.stringify(x))
                  .join('；');
              } else {
                msg = typeof d === 'string' ? d : JSON.stringify(d);
              }
            } else if (body.errmsg) {
              msg = String(body.errmsg);
            }
          }
          console.error('[request]', method, url, msg, body);
          reject(new Error(msg));
        },
        fail: (err) => {
          console.error('[request fail]', method, url, err);
          reject(err);
        },
      });
    });
  },

  /** 页面相关事件处理函数--监听用户下拉动作 */
  onPullDownRefresh() {},

  /** 页面上拉触底事件的处理函数 */
  onReachBottom() {},

  /** 用户点击右上角分享 */
  onShareAppMessage() {},

  /** 处理唤起键盘事件 */
  handleKeyboardHeightChange(event) {
    const { height } = event.detail;
    if (!height) return;
    this.setData({ keyboardHeight: height });
    wx.nextTick(this.scrollToBottom);
  },

  /** 处理收起键盘事件 */
  handleBlur() {
    this.setData({ keyboardHeight: 0 });
  },

  /** 处理输入事件 */
  handleInput(event) {
    this.setData({ input: event.detail.value });
  },

  /** 发送消息 */
  async sendMessage() {
    const { openid, student_uid, phase, input: content } = this.data;
    if (!content) return;

    const oid = (openid && String(openid).trim()) || '';
    if (!oid) {
      this._pushBot(
        '无法提交：未获取 openid。请检查服务器 WX_MINI 配置与合法域名，打开 vConsole 看 /api/wx-login 是否报错。'
      );
      wx.nextTick(this.scrollToBottom);
      return;
    }

    this.setData({ input: '' });
    this._pushUser(content);
    wx.nextTick(this.scrollToBottom);

    try {
      if (phase !== 'chatting' || !student_uid) {
        const parsed = parseStudentIdentity(content);
        if (!parsed) {
          this._pushBot('绑定完成前仅支持发送“孩子姓名 + HYDRO ID”（例如：张睿宸 37），暂不提供聊天回复。');
          wx.nextTick(this.scrollToBottom);
          return;
        }
        const r = await this._req('/api/bind-by-student-name', 'POST', {
          openid: oid,
          student_name: parsed.student_name,
          student_uid: parsed.student_uid,
          parent_name: '',
        });
        if (r && r.status === 'approved') {
          this.setData({ student_uid: r.student_uid, phase: 'chatting' });
          this._pushBot('匹配成功，已完成绑定。现在可以开始提问了。');
        } else {
          this.setData({ phase: 'pending' });
          this._pushBot('未能自动通过，已进入待审核。老师确认后即可聊天。');
        }
        wx.nextTick(this.scrollToBottom);
        return;
      }

      const res = await this._req('/api/chat', 'POST', { openid: oid, student_uid, message: content });
      const reply = (res && res.reply) ? String(res.reply) : '（未能生成回复）';
      this._pushBot(reply);
      wx.nextTick(this.scrollToBottom);
    } catch (e) {
      const tip = formatReqError(e);
      this._pushBot(tip.length > 200 ? `发送失败：${tip.slice(0, 200)}…` : `发送失败：${tip}`);
      wx.nextTick(this.scrollToBottom);
    }
  },

  /** 消息列表滚动到底部 */
  scrollToBottom() {
    this.setData({ anchor: 'bottom' });
  },
});
