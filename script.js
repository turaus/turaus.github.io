// 移动端菜单切换
const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('sidebar');

menuToggle.addEventListener('click', () => {
    sidebar.classList.toggle('active');
});

// 点击主内容区域时关闭侧边栏（移动端）
document.querySelector('.main-content').addEventListener('click', () => {
    if (window.innerWidth <= 1024) {
        sidebar.classList.remove('active');
    }
});

// 平滑滚动并更新活动导航链接
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('section');

// 导航链接点击事件
navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();

        // 移除所有活动状态
        navLinks.forEach(l => l.classList.remove('active'));

        // 添加当前活动状态
        link.classList.add('active');

        // 获取目标section
        const targetId = link.getAttribute('href');
        const targetSection = document.querySelector(targetId);

        if (targetSection) {
            // 平滑滚动到目标section
            targetSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }

        // 移动端关闭侧边栏
        if (window.innerWidth <= 1024) {
            sidebar.classList.remove('active');
        }
    });
});

// 滚动时更新导航活动状态
window.addEventListener('scroll', () => {
    let current = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;

        if (scrollY >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// 添加滚动动画
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// 观察所有模块
document.querySelectorAll('.module').forEach(module => {
    module.style.opacity = '0';
    module.style.transform = 'translateY(30px)';
    module.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(module);
});