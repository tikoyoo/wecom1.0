Page({
  onLoad() {
    // 入口页：打开小程序后直接进入聊天页（分包页）
    wx.redirectTo({
      url: '/pages/chat/index',
    });
  },
});

