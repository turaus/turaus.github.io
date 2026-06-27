// guard.js - 站点全局认证守卫（请在所有页面中引入此文件）
(function() {
    'use strict';

    // ---------- 配置项 ----------
    const AUTH_KEY = 'site_auth';          // localStorage 键名
    const EXPIRY_KEY = 'site_expiry';      // 有效期键名
    const EXPIRE_HOURS = 24;               // 有效期 24 小时（设为 0 表示永不过期）

    // ---------- 核心方法（暴露给全局） ----------
    // 检查是否已认证且未过期
    window.isSiteAuthenticated = function() {
        const status = localStorage.getItem(AUTH_KEY);
        const expiry = localStorage.getItem(EXPIRY_KEY);
        if (status !== 'true') return false;

        if (expiry) {
            const now = Date.now();
            if (now > parseInt(expiry, 10)) {
                // 已过期，清理缓存
                localStorage.removeItem(AUTH_KEY);
                localStorage.removeItem(EXPIRY_KEY);
                return false;
            }
        }
        return true;
    };

    // 设置认证状态（登录成功时调用）
    window.setSiteAuth = function() {
        localStorage.setItem(AUTH_KEY, 'true');
        if (EXPIRE_HOURS > 0) {
            const expiry = Date.now() + (EXPIRE_HOURS * 60 * 60 * 1000);
            localStorage.setItem(EXPIRY_KEY, expiry.toString());
        } else {
            localStorage.removeItem(EXPIRY_KEY); // 永不过期
        }
    };

    // 清除认证（登出用，可选）
    window.clearSiteAuth = function() {
        localStorage.removeItem(AUTH_KEY);
        localStorage.removeItem(EXPIRY_KEY);
    };

    // ---------- 路由守卫（页面加载时自动执行） ----------
    const path = window.location.pathname;
    // 判断是否为首页（根目录、/index.html、或直接 /）
    const isHome = path === '/' || path.endsWith('index.html') || path === '';

    if (!window.isSiteAuthenticated() && !isHome) {
        // 未认证 & 不在首页 → 重定向到首页
        window.location.href = '/';
    }
})();