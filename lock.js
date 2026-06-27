// lock.js - 主页密码锁逻辑（依赖 guard.js）
(function() {
    'use strict';

    // 正确密码哈希（"abc123"）
    const CORRECT_HASH = '6ca13d52ca70c883e0f0bb101e425a89e8624de51db2d2392593af6a84118090';

    const lockScreen = document.getElementById('lock-screen');
    const mainContent = document.getElementById('main-content');
    const passwordInput = document.getElementById('password-input');
    const unlockBtn = document.getElementById('unlock-btn');
    const errorMsg = document.getElementById('error-msg');

    // ---------- SHA-256 哈希函数 ----------
    async function sha256(message) {
        const msgBuffer = new TextEncoder().encode(message);
        const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    // ---------- 解锁尝试 ----------
    async function attemptUnlock() {
        const password = passwordInput.value.trim();

        if (password === '') {
            errorMsg.textContent = '⚠️ 请输入密码';
            errorMsg.classList.add('show');
            passwordInput.focus();
            return;
        }

        const inputHash = await sha256(password);

        if (inputHash === CORRECT_HASH) {
            // --- 解锁成功 ---
            // 1. 调用全局方法写入 localStorage（24小时有效）
            if (typeof window.setSiteAuth === 'function') {
                window.setSiteAuth();
            }

            // 2. 隐藏遮罩，显示内容
            lockScreen.classList.add('hidden');
            mainContent.classList.add('active');
            errorMsg.classList.remove('show');

            // 3. 清空输入框（可选）
            passwordInput.value = '';
        } else {
            // --- 解锁失败 ---
            errorMsg.textContent = '❌ 密码错误，请重试';
            errorMsg.classList.add('show');
            passwordInput.value = '';
            passwordInput.focus();

            // 抖动效果
            const card = document.querySelector('.login-card');
            card.style.animation = 'none';
            requestAnimationFrame(() => {
                card.style.animation = 'shake 0.4s ease';
            });
        }
    }

    // ---------- 事件绑定 ----------
    unlockBtn.addEventListener('click', attemptUnlock);

    passwordInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            unlockBtn.click();
        }
    });

    passwordInput.addEventListener('input', function() {
        errorMsg.classList.remove('show');
    });

    // ---------- 页面加载时检查全局登录状态 ----------
    document.addEventListener('DOMContentLoaded', function() {
        // 使用 guard.js 暴露的全局方法检查
        if (typeof window.isSiteAuthenticated === 'function' && window.isSiteAuthenticated()) {
            // 已认证 → 直接解锁
            lockScreen.classList.add('hidden');
            mainContent.classList.add('active');
        } else {
            // 未认证 → 确保锁屏可见，聚焦输入框
            lockScreen.classList.remove('hidden');
            mainContent.classList.remove('active');
            passwordInput.focus();
        }
    });

    // 点击遮罩背景聚焦输入框
    lockScreen.addEventListener('click', function(e) {
        if (e.target === lockScreen) {
            passwordInput.focus();
        }
    });
})();