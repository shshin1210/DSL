function save_figure_to_pdf(fig_handle, fn, fontsize)
if ~exist('fontsize', 'var')
    fontsize = 15;
end

if 1
    set(fig_handle,'Units','Inches');
    pos = get(fig_handle,'Position');
    set(fig_handle,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    set(findall(gcf,'-property','FontSize'),'FontSize',fontsize);
    print(fig_handle, fn,'-dpdf','-r0');
    
    % print(gcf, '-fillpage', '-dpdf', [result_path 'rmse.pdf']);
end

end