const sidebarButton = document.querySelector('.sidebar-button');
const sidebar = document.querySelector('.sidebar');

sidebarButton.addEventListener('click', () => {
    sidebarButton.classList.toggle('active');
    sidebar.classList.toggle('active');
});