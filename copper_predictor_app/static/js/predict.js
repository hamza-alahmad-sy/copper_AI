document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    try {
        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());

        // تحويل القيم إلى أرقام
        for (let key in data) {
            data[key] = parseFloat(data[key]);
        }
        
        console.log('إرسال البيانات:', data);

        // إرسال البيانات إلى الخادم
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`خطأ HTTP! الحالة: ${response.status}`);
        }

        const result = await response.json();
        console.log('نتيجة التنبؤ:', result);

        // عرض السعر المتوقع
        document.getElementById('result').textContent =
            `$${result.predicted_copper_price.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;

        // جلب معاملات النموذج من الخادم
        const coefficients = result.coefficients;  
        console.log("معاملات النموذج:", coefficients);

        // أسماء المتغيرات بنفس ترتيب النموذج
        const labels = [
            "مؤشر الطلب العالمي",
            "سعر النفط",
            "مؤشر الدولار",
            "الإنتاج الصناعي الصيني",
            "مؤشر تكاليف الطاقة",
            "معنويات السوق",
            "مؤشر انقطاع الإمدادات",
            "سعر النحاس الحالي"
        ];

        const labelKeys = [
            "global_demand_index",
            "oil_price",
            "usd_index",
            "china_industry_output",
            "energy_cost_index",
            "market_sentiment",
            "supply_disruption_index",
            "copper_price"
        ];

        // حساب التأثيرات ديناميكياً
        const impacts = labelKeys.map((key, index) => data[key] * coefficients[index]);

        console.log('المتغيرات:', labelKeys);
        console.log('التأثيرات:', impacts);

        // ألوان مختلفة لكل عمود
        const colors = [
            'rgba(255, 99, 132, 0.7)',    // أحمر - الطلب العالمي
            'rgba(54, 162, 235, 0.7)',    // أزرق - النفط
            'rgba(255, 206, 86, 0.7)',    // أصفر - الدولار
            'rgba(75, 192, 192, 0.7)',    // فيروزي - الإنتاج الصيني
            'rgba(153, 102, 255, 0.7)',   // بنفسجي - الطاقة
            'rgba(255, 159, 64, 0.7)',    // برتقالي - المشاعر
            'rgba(99, 255, 132, 0.7)',    // أخضر - انقطاع الإمدادات
            'rgba(184, 134, 11, 0.7)'     // ذهبي - سعر النحاس الحالي
        ];

        const borderColors = colors.map(c => c.replace("0.7", "1"));

        // رسم المخطط البياني
        const canvasElement = document.getElementById('impactChart');
        if (!canvasElement) {
            console.error('عنصر Canvas غير موجود!');
            return;
        }

        if (typeof Chart === 'undefined') {
            console.error('مكتبة Chart.js لم يتم تحميلها!');
            return;
        }

        const ctx = canvasElement.getContext('2d');

        if (window.impactChart && typeof window.impactChart.destroy === 'function') {
            window.impactChart.destroy();
        }

        window.impactChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'تأثير المتغير على السعر',
                    data: impacts,
                    backgroundColor: colors,
                    borderColor: borderColors,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'قيمة التأثير',
                            font: { weight: 'bold' }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            font: { size: 12, weight: 'bold' },
                            padding: 15
                        }
                    },
                    title: {
                        display: false
                    }
                }
            }
        });

        console.log('تم إنشاء المخطط بنجاح');

    } catch (error) {
        console.error('خطأ أثناء التنبؤ:', error);
        alert('❌ خطأ: ' + error.message);
    }
});
