// A full size 'lightbox' preview modal shown when left clicking on gallery previews
function closeModal() {
    gradioApp().getElementById("lightboxModal").style.display = "none";
}

function showModal(event) {
    const source = event.target || event.srcElement;
    const modalImage = gradioApp().getElementById("modalImage")
    const lb = gradioApp().getElementById("lightboxModal")
    modalImage.src = source.src
    if (modalImage.style.display === 'none') {
        lb.style.setProperty('background-image', 'url(' + source.src + ')');
    }
    lb.style.display = "flex";
    lb.focus()

    const tabTxt2Img = gradioApp().getElementById("tab_txt2img")
    const tabImg2Img = gradioApp().getElementById("tab_img2img")
    // show the save button in modal only on txt2img or img2img tabs
    if (tabTxt2Img.style.display != "none" || tabImg2Img.style.display != "none") {
        gradioApp().getElementById("modal_save").style.display = "inline"
    } else {
        gradioApp().getElementById("modal_save").style.display = "none"
    }
    event.stopPropagation()
}

function negmod(n, m) {
    return ((n % m) + m) % m;
}

function updateOnBackgroundChange() {
    const modalImage = gradioApp().getElementById("modalImage")
    if (modalImage && modalImage.offsetParent) {
        var buttons = all_gallery_buttons();
        var result = -1;
        buttons.forEach(function(v, i){ if(v.classList.contains('selected')) { result = i } })

        if (result >= 0 && result <buttons.length) {
            let currentButton = buttons[result];
            if (currentButton?.children?.length > 0 && modalImage.src != currentButton.children[0].src) {
                modalImage.src = currentButton.children[0].src;
                if (modalImage.style.display === 'none') {
                    modal.style.setProperty('background-image', `url(${modalImage.src})`)
                }
            }
        }
    }
}

function modalImageSwitch(offset) {
    var tabs = gradioApp().querySelector('#tabs').querySelectorAll('button')
    var index = tabs[0].className.indexOf('selected') != -1 ? 0 : 1
    var allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem')[index].querySelectorAll('.thumbnail-item.thumbnail-small');
    var galleryButtons = []
    allGalleryButtons.forEach(function(elem) {
        if (elem.parentElement.offsetParent) {
            galleryButtons.push(elem);
        }
    })

    if (galleryButtons.length > 1) {
        var result = -1

        allGalleryButtons.forEach(function(v, i){ if(v.classList.contains('selected')) { result = i } })

        if (result != -1) {
            nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)]
            nextButton.click()
            const modalImage = gradioApp().getElementById("modalImage");
            const modal = gradioApp().getElementById("lightboxModal");
            modalImage.src = nextButton.children[0].src;
            if (modalImage.style.display === 'none') {
                modal.style.setProperty('background-image', `url(${modalImage.src})`)
            }
            setTimeout(function() {
                modal.focus()
            }, 10)
        }
    }
}

function saveImage(){
    const tabTxt2Img = gradioApp().getElementById("tab_txt2img")
    const tabImg2Img = gradioApp().getElementById("tab_img2img")
    const saveTxt2Img = "save_txt2img"
    const saveImg2Img = "save_img2img"
    if (tabTxt2Img.style.display != "none") {
        gradioApp().getElementById(saveTxt2Img).click()
    } else if (tabImg2Img.style.display != "none") {
        gradioApp().getElementById(saveImg2Img).click()
    } else {
        console.error("missing implementation for saving modal of this type")
    }
}

function modalSaveImage(event) {
    saveImage()
    event.stopPropagation()
}

function modalNextImage(event) {
    modalImageSwitch(1)
    event.stopPropagation()
}

function modalPrevImage(event) {
    modalImageSwitch(-1)
    event.stopPropagation()
}

function modalKeyHandler(event) {
    switch (event.key) {
        case "s":
            saveImage()
            break;
        case "ArrowLeft":
            modalPrevImage(event)
            break;
        case "ArrowRight":
            modalNextImage(event)
            break;
        case "Escape":
            closeModal();
            break;
    }
}

function showGalleryImage() {
    setTimeout(function() {
        fullImg_preview = gradioApp().querySelectorAll('.gradio-gallery > div > img')

        if (fullImg_preview != null) {
            fullImg_preview.forEach(function function_name(e) {
                if (e.dataset.modded)
                    return;
                e.dataset.modded = true;
                if(e && e.parentElement.tagName == 'DIV'){
                    e.style.cursor='pointer'
                    e.style.userSelect='none'
                    e.addEventListener('click', function (evt) {
                        if(!opts.js_modal_lightbox) return;
                        modalZoomSet(gradioApp().getElementById('modalImage'), opts.js_modal_lightbox_initially_zoomed)
                        showModal(evt)
                    }, true);
                }
            });
        }

    }, 100);
}

function modalZoomSet(modalImage, enable) {
    if (enable) {
        modalImage.classList.add('modalImageFullscreen');
    } else {
        modalImage.classList.remove('modalImageFullscreen');
    }
}

function modalZoomToggle(event) {
    modalImage = gradioApp().getElementById("modalImage");
    modalZoomSet(modalImage, !modalImage.classList.contains('modalImageFullscreen'))
    event.stopPropagation()
}

function modalTileImageToggle(event) {
    const modalImage = gradioApp().getElementById("modalImage");
    const modal = gradioApp().getElementById("lightboxModal");
    const isTiling = modalImage.style.display === 'none';
    if (isTiling) {
        modalImage.style.display = 'block';
        modal.style.setProperty('background-image', 'none')
    } else {
        modalImage.style.display = 'none';
        modal.style.setProperty('background-image', `url(${modalImage.src})`)
    }

    event.stopPropagation()
}

function galleryImageHandler(e) {
    e.onclick = showGalleryImage;
}

onUiUpdate(function() {
    fullImg_preview = gradioApp().querySelectorAll('.gradio-gallery > div > img')
    if (fullImg_preview != null) {
        fullImg_preview.forEach(galleryImageHandler);
    }
    updateOnBackgroundChange();
})

document.addEventListener("DOMContentLoaded", function() {
    const modalFragment = document.createDocumentFragment();
    const modal = document.createElement('div')
    modal.onclick = closeModal;
    modal.id = "lightboxModal";
    modal.tabIndex = 0
    modal.addEventListener('keydown', modalKeyHandler, true)

    const modalControls = document.createElement('div')
    modalControls.className = 'modalControls gradio-container';
    modal.append(modalControls);

    const modalZoom = document.createElement('span')
    modalZoom.className = 'modalZoom cursor';
    modalZoom.innerHTML = '&#10529;'
    modalZoom.addEventListener('click', modalZoomToggle, true)
    modalZoom.title = "Toggle zoomed view";
    modalControls.appendChild(modalZoom)

    const modalTileImage = document.createElement('span')
    modalTileImage.className = 'modalTileImage cursor';
    modalTileImage.innerHTML = '&#8862;'
    modalTileImage.addEventListener('click', modalTileImageToggle, true)
    modalTileImage.title = "Preview tiling";
    modalControls.appendChild(modalTileImage)

    const modalSave = document.createElement("span")
    modalSave.className = "modalSave cursor"
    modalSave.id = "modal_save"
    modalSave.innerHTML = "&#x1F5AB;"
    modalSave.addEventListener("click", modalSaveImage, true)
    modalSave.title = "Save Image(s)"
    modalControls.appendChild(modalSave)

    const modalClose = document.createElement('span')
    modalClose.className = 'modalClose cursor';
    modalClose.innerHTML = '&times;'
    modalClose.onclick = closeModal;
    modalClose.title = "Close image viewer";
    modalControls.appendChild(modalClose)

    const modalImage = document.createElement('img')
    modalImage.id = 'modalImage';
    modalImage.onclick = closeModal;
    modalImage.tabIndex = 0
    modalImage.addEventListener('keydown', modalKeyHandler, true)
    modal.appendChild(modalImage)

    const modalPrev = document.createElement('a')
    modalPrev.className = 'modalPrev';
    modalPrev.innerHTML = '&#10094;'
    modalPrev.tabIndex = 0
    modalPrev.addEventListener('click', modalPrevImage, true);
    modalPrev.addEventListener('keydown', modalKeyHandler, true)
    modal.appendChild(modalPrev)

    const modalNext = document.createElement('a')
    modalNext.className = 'modalNext';
    modalNext.innerHTML = '&#10095;'
    modalNext.tabIndex = 0
    modalNext.addEventListener('click', modalNextImage, true);
    modalNext.addEventListener('keydown', modalKeyHandler, true)

    modal.appendChild(modalNext)

    gradioApp().appendChild(modal)

    document.body.appendChild(modal);
});
