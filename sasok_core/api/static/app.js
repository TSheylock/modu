/**
 * SASOK - Визуализация семантической сети
 * Модуль для работы с графом знаний и эмоциональным контекстом
 * Использует архетипические слои и метакогнитивные механизмы
 */

// Константы API
const API_BASE_URL = '/api/graph';

// Инициализация Cytoscape
const cy = cytoscape({
    container: document.getElementById('cy'),
    style: [
        {
            selector: 'node',
            style: {
                'background-color': 'data(color)',
                'label': 'data(name)',
                'color': '#000',
                'text-outline-width': 2,
                'text-outline-color': '#fff',
                'font-size': 12,
                'text-valign': 'center',
                'text-halign': 'center',
                'width': 30,
                'height': 30
            }
        },
        {
            selector: 'edge',
            style: {
                'width': 'data(weight)',
                'line-color': '#999',
                'target-arrow-color': '#999',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'label': 'data(label)',
                'font-size': 10,
                'text-rotation': 'autorotate',
                'text-outline-width': 2,
                'text-outline-color': '#fff'
            }
        },
        {
            selector: ':selected',
            style: {
                'border-width': 3,
                'border-color': '#369'
            }
        }
    ],
    layout: {
        name: 'cose',
        padding: 30,
        animate: false,
        fit: true
    }
});

// Загрузка данных графа
async function loadGraphData(params = {}) {
    showLoading(true);
    
    try {
        // Создание строки запроса
        const queryParams = new URLSearchParams();
        for (const [key, value] of Object.entries(params)) {
            if (value) queryParams.append(key, value);
        }
        
        const response = await fetch(`${API_BASE_URL}/data?${queryParams.toString()}`);
        const data = await response.json();
        
        // Очистка и добавление элементов
        cy.elements().remove();
        cy.add(data.nodes);
        cy.add(data.edges);
        
        // Применение компактного макета
        cy.layout({
            name: 'cose',
            padding: 30,
            animate: false,
            fit: true,
            componentSpacing: 40,
            nodeOverlap: 20,
            idealEdgeLength: 100,
            edgeElasticity: 100
        }).run();
        
        updateStatusMessage(`Загружено ${data.nodes.length} узлов и ${data.edges.length} связей`);
    } catch (error) {
        console.error('Ошибка загрузки графа:', error);
        updateStatusMessage('Ошибка загрузки графа');
    } finally {
        showLoading(false);
    }
}

// Загрузка списка концептов
async function loadConcepts() {
    try {
        const response = await fetch(`${API_BASE_URL}/concepts`);
        const data = await response.json();
        
        const selectElement = document.getElementById('search-concept');
        selectElement.innerHTML = '<option value="">Выберите концепт...</option>';
        
        data.concepts.forEach(concept => {
            const option = document.createElement('option');
            option.value = concept;
            option.textContent = concept;
            selectElement.appendChild(option);
        });
    } catch (error) {
        console.error('Ошибка загрузки концептов:', error);
    }
}

// Загрузка списка эмоций
async function loadEmotions() {
    try {
        const response = await fetch(`${API_BASE_URL}/emotions`);
        const data = await response.json();
        
        const selectElement = document.getElementById('search-emotion');
        selectElement.innerHTML = '<option value="">Выберите эмоцию...</option>';
        
        // Базовые эмоции
        if (data.basic && data.basic.length > 0) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = 'Базовые эмоции';
            
            data.basic.forEach(emotion => {
                const option = document.createElement('option');
                option.value = emotion;
                option.textContent = emotion;
                optgroup.appendChild(option);
            });
            
            selectElement.appendChild(optgroup);
        }
        
        // Сложные эмоции
        if (data.complex && data.complex.length > 0) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = 'Сложные эмоции';
            
            data.complex.forEach(emotion => {
                const option = document.createElement('option');
                option.value = emotion;
                option.textContent = emotion;
                optgroup.appendChild(option);
            });
            
            selectElement.appendChild(optgroup);
        }
        
        // Производные эмоции
        if (data.derived && data.derived.length > 0) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = 'Производные эмоции';
            
            data.derived.forEach(emotion => {
                const option = document.createElement('option');
                option.value = emotion;
                option.textContent = emotion;
                optgroup.appendChild(option);
            });
            
            selectElement.appendChild(optgroup);
        }
    } catch (error) {
        console.error('Ошибка загрузки эмоций:', error);
    }
}

// Загрузка эмоционального профиля
async function loadEmotionProfile(concept) {
    try {
        const response = await fetch(`${API_BASE_URL}/emotion-profile/${concept}`);
        const data = await response.json();
        
        const container = document.getElementById('emotion-profile');
        container.innerHTML = '';
        
        const emotions = data.emotions;
        const keys = Object.keys(emotions);
        
        if (keys.length === 0) {
            container.innerHTML = '<p>Нет данных об эмоциональном профиле</p>';
            return;
        }
        
        for (const emotion of keys) {
            const item = document.createElement('div');
            item.className = 'emotion-profile-item';
            
            const nameSpan = document.createElement('span');
            nameSpan.textContent = emotion;
            
            const valueSpan = document.createElement('span');
            valueSpan.textContent = (emotions[emotion].weight * 100).toFixed(0) + '%';
            
            item.appendChild(nameSpan);
            item.appendChild(valueSpan);
            
            const progressDiv = document.createElement('div');
            progressDiv.className = 'progress';
            
            const progressBar = document.createElement('div');
            progressBar.className = 'progress-bar';
            
            // Выбор цвета в зависимости от типа эмоции
            if (emotions[emotion].type === 'basic') {
                progressBar.className += ' bg-danger';
            } else if (emotions[emotion].type === 'complex') {
                progressBar.className += ' bg-warning';
            } else {
                progressBar.className += ' bg-info';
            }
            
            progressBar.style.width = (emotions[emotion].weight * 100) + '%';
            progressBar.setAttribute('aria-valuenow', emotions[emotion].weight * 100);
            progressBar.setAttribute('aria-valuemin', '0');
            progressBar.setAttribute('aria-valuemax', '100');
            
            progressDiv.appendChild(progressBar);
            
            container.appendChild(item);
            container.appendChild(progressDiv);
        }
        
        document.getElementById('emotion-profile-container').style.display = 'block';
    } catch (error) {
        console.error('Ошибка загрузки эмоционального профиля:', error);
    }
}

// Загрузка похожих концептов
async function loadSimilarConcepts(concept) {
    try {
        const response = await fetch(`${API_BASE_URL}/similar-concepts/${concept}`);
        const data = await response.json();
        
        const container = document.getElementById('similar-concepts');
        container.innerHTML = '';
        
        const similar = data.similar_concepts;
        
        if (similar.length === 0) {
            container.innerHTML = '<p>Нет данных о похожих концептах</p>';
            return;
        }
        
        const list = document.createElement('ul');
        list.className = 'list-group';
        
        for (const item of similar) {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            li.textContent = item.name;
            
            const badge = document.createElement('span');
            badge.className = 'badge bg-primary rounded-pill';
            badge.textContent = (item.similarity_score * 100).toFixed(0) + '%';
            
            li.appendChild(badge);
            list.appendChild(li);
            
            // Добавление обработчика события
            li.style.cursor = 'pointer';
            li.onclick = function() {
                document.getElementById('search-concept').value = item.name;
                showConcept(item.name);
            };
        }
        
        container.appendChild(list);
        document.getElementById('similar-concepts-container').style.display = 'block';
    } catch (error) {
        console.error('Ошибка загрузки похожих концептов:', error);
    }
}

// Показать концепт в графе
function showConcept(concept) {
    const depth = document.getElementById('depth-range').value;
    loadGraphData({ start_node: concept, node_type: 'Concept', depth });
    loadEmotionProfile(concept);
    loadSimilarConcepts(concept);
    updateStatusMessage(`Показан концепт: ${concept}`);
}

// Показать эмоцию в графе
function showEmotion(emotion) {
    const depth = document.getElementById('depth-range').value;
    loadGraphData({ start_node: emotion, node_type: 'Emotion', depth });
    document.getElementById('emotion-profile-container').style.display = 'none';
    document.getElementById('similar-concepts-container').style.display = 'none';
    updateStatusMessage(`Показана эмоция: ${emotion}`);
}

// Показать информацию о узле
function showNodeInfo(node) {
    const container = document.getElementById('node-details');
    container.innerHTML = '';
    
    const data = node.data();
    
    // Заголовок с типом узла
    const title = document.createElement('h6');
    let typeBadge = '';
    
    if (data.label === 'Emotion') {
        typeBadge = '<span class="badge-emotion">Эмоция</span>';
    } else if (data.label === 'Concept') {
        typeBadge = '<span class="badge-concept">Концепт</span>';
    } else if (data.label === 'Property') {
        typeBadge = '<span class="badge-property">Свойство</span>';
    }
    
    title.innerHTML = `${data.name} ${typeBadge}`;
    container.appendChild(title);
    
    // Свойства узла
    const propList = document.createElement('dl');
    propList.className = 'row';
    
    for (const [key, value] of Object.entries(data.properties)) {
        if (key === 'name') continue; // Имя уже отображается в заголовке
        
        const dt = document.createElement('dt');
        dt.className = 'col-sm-4';
        dt.textContent = key;
        
        const dd = document.createElement('dd');
        dd.className = 'col-sm-8';
        dd.textContent = value;
        
        propList.appendChild(dt);
        propList.appendChild(dd);
    }
    
    container.appendChild(propList);
    
    // Кнопки действий
    const actionDiv = document.createElement('div');
    actionDiv.className = 'mt-3';
    
    if (data.label === 'Concept') {
        const profileBtn = document.createElement('button');
        profileBtn.className = 'btn btn-sm btn-outline-primary me-2';
        profileBtn.textContent = 'Эмоциональный профиль';
        profileBtn.onclick = function() {
            loadEmotionProfile(data.name);
        };
        
        const similarBtn = document.createElement('button');
        similarBtn.className = 'btn btn-sm btn-outline-info';
        similarBtn.textContent = 'Похожие концепты';
        similarBtn.onclick = function() {
            loadSimilarConcepts(data.name);
        };
        
        actionDiv.appendChild(profileBtn);
        actionDiv.appendChild(similarBtn);
    }
    
    container.appendChild(actionDiv);
    
    document.getElementById('node-info-container').style.display = 'block';
}

// Управление индикатором загрузки
function showLoading(show) {
    document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
}

// Обновление сообщения в строке состояния
function updateStatusMessage(message) {
    document.getElementById('status-message').textContent = message;
}

// Инициализация приложения
async function initApp() {
    showLoading(true);
    
    try {
        // Проверка состояния графа
        const response = await fetch(`${API_BASE_URL}/`);
        const status = await response.json();
        
        if (status.status === 'ok') {
            updateStatusMessage(`Граф знаний активен: ${status.nodes} узлов, ${status.relationships} связей`);
            
            // Загрузка данных
            await loadConcepts();
            await loadEmotions();
            await loadGraphData({ depth: 1 });
        } else {
            updateStatusMessage(`Ошибка: ${status.message}`);
        }
    } catch (error) {
        console.error('Ошибка инициализации приложения:', error);
        updateStatusMessage('Ошибка подключения к API');
    } finally {
        showLoading(false);
    }
}

// Обработчики событий
document.getElementById('btn-reload').addEventListener('click', function() {
    const depth = document.getElementById('depth-range').value;
    loadGraphData({ depth });
    document.getElementById('emotion-profile-container').style.display = 'none';
    document.getElementById('similar-concepts-container').style.display = 'none';
    document.getElementById('node-info-container').style.display = 'none';
});

document.getElementById('btn-search-concept').addEventListener('click', function() {
    const concept = document.getElementById('search-concept').value;
    if (concept) {
        showConcept(concept);
    }
});

document.getElementById('btn-search-emotion').addEventListener('click', function() {
    const emotion = document.getElementById('search-emotion').value;
    if (emotion) {
        showEmotion(emotion);
    }
});

document.getElementById('depth-range').addEventListener('input', function() {
    document.getElementById('depth-value').textContent = this.value;
});

// Обработка событий взаимодействия с графом
cy.on('tap', 'node', function(event) {
    const node = event.target;
    showNodeInfo(node);
});

// Обработка архетипического шаблона
cy.on('mouseover', 'node', function(event) {
    const node = event.target;
    const data = node.data();
    
    // Пометка узла для пользователя - метакогнитивный механизм
    if (data.label === 'Emotion') {
        node.style('border-width', 2);
        node.style('border-color', '#FF5733');
    }
});

cy.on('mouseout', 'node', function(event) {
    const node = event.target;
    // Возврат к нормальному состоянию
    node.style('border-width', 0);
});

// Запуск приложения
initApp();
