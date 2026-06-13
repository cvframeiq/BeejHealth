import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

export default function HomePageI18n({ nav, setAuth, setAuthMode }) {
  const { t } = useTranslation();
  const [contactForm, setContactForm] = useState({
    name: '',
    mobile: '',
    email: '',
    company: '',
    subject: '',
    message: '',
    type: 'farmer',
  });
  const [contactDone, setContactDone] = useState(false);
  const [faqOpen, setFaqOpen] = useState(null);

  const hero = t('home.hero', { returnObjects: true });
  const achievements = t('home.achievements', { returnObjects: true });
  const howItWorks = t('home.howItWorks', { returnObjects: true });
  const features = t('home.features', { returnObjects: true });
  const testimonials = t('home.testimonials', { returnObjects: true });
  const crops = t('home.crops', { returnObjects: true });
  const team = t('home.team', { returnObjects: true });
  const faqs = t('home.faqs', { returnObjects: true });
  const contact = t('home.contact', { returnObjects: true });
  const cta = t('home.cta', { returnObjects: true });
  const footer = t('home.footer', { returnObjects: true });

  const submitContact = () => {
    if (!contactForm.name.trim() || !contactForm.message.trim()) {
      window.alert(contact.alertRequired);
      return;
    }
    setContactDone(true);
  };

  return (
    <>
      <section className="hero">
        <div style={{ position: 'absolute', width: 600, height: 600, background: 'radial-gradient(circle,rgba(77,189,122,.1),transparent)', top: -100, right: -80, borderRadius: '50%', pointerEvents: 'none' }} />
        <div style={{ position: 'absolute', width: 350, height: 350, background: 'radial-gradient(circle,rgba(26,111,212,.07),transparent)', bottom: -60, left: -40, borderRadius: '50%', pointerEvents: 'none' }} />
        <div className="hero-in">
          <div style={{ animation: 'slideUp .5s ease' }}>
            <div className="hero-pill" style={{ marginBottom: 18 }}>
              <div className="hero-dot" />🏆 {hero.badge}
            </div>
            <h1 className="hero-h1">
              {hero.titleLine1}
              <br />
              <em>{hero.titleLine2}</em> 🌱
            </h1>
            <p className="hero-p">
              {hero.description} <strong style={{ color: 'var(--g3)' }}>{hero.descriptionStrong}</strong>
            </p>
            <div className="hero-btns" style={{ marginBottom: 28 }}>
              <button className="btn btn-g btn-xl" onClick={() => nav('consultation')}>🔬 {hero.primaryCta}</button>
              <button className="btn btn-out btn-xl" onClick={() => nav('experts')}>👨‍⚕️ {hero.secondaryCta}</button>
            </div>
            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 28 }}>
              {hero.trustBadges.map((badge) => (
                <div key={badge} style={{ fontSize: 12, fontWeight: 700, color: 'var(--tx3)', background: 'white', padding: '5px 12px', borderRadius: 100, border: '1px solid var(--br)', boxShadow: '0 1px 4px rgba(0,0,0,.04)' }}>
                  {badge}
                </div>
              ))}
            </div>
            <div className="stats-row">
              {hero.stats.map((stat) => (
                <div key={stat.label}>
                  <div className="stat-n">{stat.value}</div>
                  <div className="stat-l">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="hero-card" style={{ animation: 'slideUp .55s .1s ease both' }}>
            <div className="hc-lbl">🤖 {hero.demo.label}</div>
            <div className="dis-card">
              <div className="dc-crop">🥥 {hero.demo.crop}</div>
              <div className="dc-name">{hero.demo.title}</div>
              <div className="dc-sci">{hero.demo.subtitle}</div>
              <div className="dc-bar-row"><span>{hero.demo.confidence}</span><span style={{ color: 'var(--r2)', fontWeight: 800 }}>90.2%</span></div>
              <div className="dc-bar"><div className="dc-fill" style={{ width: '90%', background: 'var(--r3)' }} /></div>
              <div className="dc-bar-row" style={{ marginTop: 6 }}><span>{hero.demo.urgency}</span><span style={{ color: 'var(--a1)', fontWeight: 800 }}>{hero.demo.urgencyValue}</span></div>
              <div className="dc-bar"><div className="dc-fill" style={{ width: '80%', background: 'var(--a3)' }} /></div>
              <div style={{ display: 'flex', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
                {hero.demo.tags.map((tag, index) => (
                  <span key={tag} style={{ padding: '4px 10px', background: index === 0 ? 'var(--rp)' : index === 1 ? 'var(--bp)' : 'var(--gp)', borderRadius: 100, fontSize: 11, fontWeight: 700, color: index === 0 ? 'var(--r2)' : index === 1 ? 'var(--b2)' : 'var(--g3)' }}>
                    {tag}
                  </span>
                ))}
              </div>
              <div className="dc-pill" style={{ marginTop: 10 }}>💊 {hero.demo.medicine}</div>
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <button className="btn btn-g btn-sm" style={{ flex: 1 }} onClick={() => nav('consultation')}>🔬 {hero.demo.primaryCta}</button>
              <button className="btn btn-out btn-sm" style={{ flex: 1 }} onClick={() => nav('experts')}>👨‍⚕️ {hero.demo.secondaryCta}</button>
            </div>
            <div style={{ marginTop: 10, padding: '8px 12px', background: 'var(--gp)', borderRadius: 8, fontSize: 12, color: 'var(--tx3)', display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--g4)', flexShrink: 0 }} />
              <span>{hero.demo.liveStatus}</span>
            </div>
          </div>
        </div>
      </section>

      <section style={{ padding: '32px 28px', background: 'white', borderBottom: '1px solid var(--br)' }}>
        <div style={{ maxWidth: 1160, margin: '0 auto' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8,1fr)', gap: 8 }}>
            {achievements.map((item) => (
              <div key={item.label} style={{ textAlign: 'center', padding: '12px 4px' }}>
                <div style={{ fontSize: 20, marginBottom: 4 }}>{item.icon}</div>
                <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 20, fontWeight: 900, color: 'var(--g1)' }}>{item.value}</div>
                <div style={{ fontSize: 10, color: 'var(--tx3)', fontWeight: 600, marginTop: 2, lineHeight: 1.3 }}>{item.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section style={{ padding: '72px 28px', background: 'var(--gb)' }}>
        <div style={{ maxWidth: 1160, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 48 }}>
            <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 34, fontWeight: 900, color: 'var(--g1)', marginBottom: 8 }}>⚡ {howItWorks.title}</div>
            <div style={{ fontSize: 15, color: 'var(--tx2)' }}>{howItWorks.subtitle}</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 28, position: 'relative' }}>
            <div style={{ position: 'absolute', top: 52, left: '16%', right: '16%', height: 2, background: 'linear-gradient(90deg,var(--g4),var(--b3))', zIndex: 0, borderRadius: 2 }} />
            {howItWorks.steps.map((step) => (
              <div key={step.number} className="card" style={{ padding: 28, position: 'relative', zIndex: 1 }}>
                <div style={{ width: 52, height: 52, borderRadius: 14, background: step.bg, border: `2px solid ${step.color}44`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24, marginBottom: 16 }}>{step.icon}</div>
                <div style={{ position: 'absolute', top: 18, right: 20, fontFamily: "'Baloo 2',cursive", fontSize: 44, fontWeight: 900, color: 'var(--br)', lineHeight: 1 }}>{step.number}</div>
                <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--tx)', marginBottom: 8 }}>{step.title}</div>
                <div style={{ fontSize: 13.5, color: 'var(--tx2)', lineHeight: 1.7 }}>{step.description}</div>
              </div>
            ))}
          </div>
          <div style={{ textAlign: 'center', marginTop: 32 }}>
            <button className="btn btn-g btn-xl" onClick={() => nav('consultation')}>🚀 {howItWorks.cta}</button>
          </div>
        </div>
      </section>

      <section style={{ padding: '72px 28px', background: 'white' }}>
        <div style={{ maxWidth: 1160, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 48 }}>
            <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 34, fontWeight: 900, color: 'var(--g1)', marginBottom: 8 }}>🚀 {features.title}</div>
            <div style={{ fontSize: 15, color: 'var(--tx2)' }}>{features.subtitle}</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 20 }}>
            {features.items.map((feature) => (
              <div key={feature.title} className="card card-hov" style={{ padding: 22, background: feature.bg, border: 'none', position: 'relative' }}>
                <div style={{ position: 'absolute', top: 14, right: 14, fontSize: 10, fontWeight: 800, padding: '3px 9px', borderRadius: 100, background: 'rgba(255,255,255,.8)', color: feature.tc, border: `1px solid ${feature.tc}33` }}>{feature.tag}</div>
                <div style={{ fontSize: 30, marginBottom: 12 }}>{feature.icon}</div>
                <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--tx)', marginBottom: 7 }}>{feature.title}</div>
                <div style={{ fontSize: 13, color: 'var(--tx2)', lineHeight: 1.65 }}>{feature.description}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section style={{ padding: '72px 28px', background: 'var(--gb)' }}>
        <div style={{ maxWidth: 1160, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 44 }}>
            <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 34, fontWeight: 900, color: 'var(--g1)', marginBottom: 8 }}>❤️ {testimonials.title}</div>
            <div style={{ fontSize: 15, color: 'var(--tx2)' }}>{testimonials.subtitle}</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 20 }}>
            {testimonials.items.map((item) => (
              <div key={item.name} className="card" style={{ padding: 22, borderTop: '3px solid var(--g4)' }}>
                <div style={{ display: 'flex', gap: 2, marginBottom: 10 }}>
                  {[...Array(5)].map((_, index) => <span key={index} style={{ color: 'var(--a2)', fontSize: 14 }}>★</span>)}
                </div>
                <div style={{ fontSize: 13.5, color: 'var(--tx)', lineHeight: 1.72, marginBottom: 16, fontStyle: 'italic' }}>"{item.text}"</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: 800, fontSize: 14, color: 'var(--tx)' }}>{item.name}</div>
                    <div style={{ fontSize: 12, color: 'var(--tx3)' }}>{item.crop} • {item.location}</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: 10, color: 'var(--tx3)', marginBottom: 2 }}>{item.savedLabel}</div>
                    <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 18, fontWeight: 900, color: 'var(--g4)' }}>{item.savedValue}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section style={{ padding: '56px 28px', background: 'white' }}>
        <div style={{ maxWidth: 1160, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 32 }}>
            <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 30, fontWeight: 900, color: 'var(--g1)', marginBottom: 6 }}>🌾 {crops.title}</div>
            <div style={{ fontSize: 14, color: 'var(--tx2)' }}>{crops.subtitle}</div>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, justifyContent: 'center' }}>
            {crops.items.map((crop) => (
              <div key={crop.name} className="card card-hov" style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', minWidth: 140, border: crop.tag ? '1.5px solid var(--g4)' : '1.5px solid var(--br)' }} onClick={() => nav('consultation')}>
                <span style={{ fontSize: 24 }}>{crop.icon}</span>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--tx)', display: 'flex', alignItems: 'center', gap: 5 }}>
                    {crop.name}
                    {crop.tag ? <span style={{ fontSize: 9, background: 'var(--g4)', color: 'white', padding: '1px 6px', borderRadius: 100, fontWeight: 700 }}>{crop.tag}</span> : null}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--tx3)' }}>{crop.count}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section style={{ padding: '72px 28px', background: 'var(--gb)' }}>
        <div style={{ maxWidth: 1160, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 44 }}>
            <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 34, fontWeight: 900, color: 'var(--g1)', marginBottom: 8 }}>👥 {team.title}</div>
            <div style={{ fontSize: 15, color: 'var(--tx2)' }}>{team.subtitle}</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 22 }}>
            {team.items.map((member) => (
              <div key={member.name} className="card" style={{ padding: 24, textAlign: 'center' }}>
                <div style={{ width: 68, height: 68, borderRadius: '50%', background: member.bg, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontSize: 22, fontWeight: 900, margin: '0 auto 14px' }}>{member.initials}</div>
                <div style={{ fontWeight: 800, fontSize: 15, color: 'var(--tx)', marginBottom: 3 }}>{member.name}</div>
                <div style={{ fontSize: 12, color: 'var(--g4)', fontWeight: 700, marginBottom: 8 }}>{member.role}</div>
                <div style={{ fontSize: 12, color: 'var(--tx2)', lineHeight: 1.6 }}>{member.description}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section style={{ padding: '72px 28px', background: 'white' }}>
        <div style={{ maxWidth: 760, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 44 }}>
            <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 34, fontWeight: 900, color: 'var(--g1)', marginBottom: 8 }}>❓ {faqs.title}</div>
            <div style={{ fontSize: 15, color: 'var(--tx2)' }}>{faqs.subtitle}</div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {faqs.items.map((faq, index) => (
              <div key={faq.question} className="card" style={{ overflow: 'hidden', cursor: 'pointer' }} onClick={() => setFaqOpen(faqOpen === index ? null : index)}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '15px 20px' }}>
                  <div style={{ fontWeight: 700, fontSize: 14.5, color: 'var(--tx)', paddingRight: 16 }}>{faq.question}</div>
                  <div style={{ fontSize: 20, color: 'var(--g4)', flexShrink: 0, transform: faqOpen === index ? 'rotate(45deg)' : 'none', transition: 'transform .2s', lineHeight: 1 }}>+</div>
                </div>
                {faqOpen === index ? <div style={{ padding: '0 20px 16px', fontSize: 13.5, color: 'var(--tx2)', lineHeight: 1.72, borderTop: '1px solid var(--br)', paddingTop: 12 }}>{faq.answer}</div> : null}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section style={{ padding: '72px 28px', background: 'var(--gb)' }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 48, alignItems: 'start' }}>
          <div>
            <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 34, fontWeight: 900, color: 'var(--g1)', marginBottom: 10 }}>📩 {contact.title}</div>
            <div style={{ fontSize: 15, color: 'var(--tx2)', lineHeight: 1.75, marginBottom: 28 }}>{contact.subtitle}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
              {contact.info.map((item) => (
                <div key={item.title} style={{ display: 'flex', gap: 14, alignItems: 'flex-start' }}>
                  <div style={{ width: 40, height: 40, borderRadius: 10, background: 'var(--gp)', border: '1px solid var(--br)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18, flexShrink: 0 }}>{item.icon}</div>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--tx3)', textTransform: 'uppercase', letterSpacing: 0.4, marginBottom: 2 }}>{item.title}</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--tx)' }}>{item.value}</div>
                  </div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 24 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--tx3)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 0.4 }}>{contact.socialLabel}</div>
              <div style={{ display: 'flex', gap: 9 }}>
                {['📷', '🐦', '💼', '📘', '▶️'].map((icon) => <div key={icon} style={{ width: 38, height: 38, borderRadius: 9, background: 'white', border: '1px solid var(--br)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: 16 }}>{icon}</div>)}
              </div>
            </div>
          </div>

          <div className="card" style={{ padding: 28 }}>
            {contactDone ? (
              <div style={{ textAlign: 'center', padding: '36px 16px' }}>
                <div style={{ fontSize: 52, marginBottom: 14 }}>🎉</div>
                <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 22, fontWeight: 900, color: 'var(--g1)', marginBottom: 8 }}>{contact.doneTitle}</div>
                <div style={{ fontSize: 14, color: 'var(--tx2)', lineHeight: 1.7 }}>{contact.doneText}</div>
                <button className="btn btn-g btn-md" style={{ marginTop: 18 }} onClick={() => setContactDone(false)}>{contact.doneAction}</button>
              </div>
            ) : (
              <>
                <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 20, fontWeight: 900, color: 'var(--g1)', marginBottom: 18 }}>✉️ {contact.formTitle}</div>
                <div className="fgrp">
                  <label className="flbl">{contact.typeLabel}</label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8 }}>
                    {contact.typeOptions.map((option) => (
                      <button key={option.value} onClick={() => setContactForm((prev) => ({ ...prev, type: option.value }))} style={{ padding: '9px 6px', borderRadius: 9, fontSize: 12, fontWeight: 700, border: `2px solid ${contactForm.type === option.value ? 'var(--g4)' : 'var(--br)'}`, background: contactForm.type === option.value ? 'var(--gp)' : 'white', color: contactForm.type === option.value ? 'var(--g3)' : 'var(--tx2)', cursor: 'pointer' }}>
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="frow">
                  <div className="fgrp">
                    <label className="flbl">{contact.name}</label>
                    <input className="finp" placeholder={contact.namePlaceholder} value={contactForm.name} onChange={(e) => setContactForm((prev) => ({ ...prev, name: e.target.value }))} />
                  </div>
                  <div className="fgrp">
                    <label className="flbl">{contact.mobile}</label>
                    <input className="finp" placeholder={contact.mobilePlaceholder} value={contactForm.mobile} maxLength={10} onChange={(e) => setContactForm((prev) => ({ ...prev, mobile: e.target.value.replace(/[^0-9]/g, '') }))} />
                  </div>
                </div>
                <div className="frow">
                  <div className="fgrp">
                    <label className="flbl">{contact.email}</label>
                    <input className="finp" type="email" placeholder={contact.emailPlaceholder} value={contactForm.email} onChange={(e) => setContactForm((prev) => ({ ...prev, email: e.target.value }))} />
                  </div>
                  <div className="fgrp">
                    <label className="flbl">{contact.company}</label>
                    <input className="finp" placeholder={contact.companyPlaceholder} value={contactForm.company} onChange={(e) => setContactForm((prev) => ({ ...prev, company: e.target.value }))} />
                  </div>
                </div>
                <div className="fgrp">
                  <label className="flbl">{contact.subject}</label>
                  <select className="fsel" value={contactForm.subject} onChange={(e) => setContactForm((prev) => ({ ...prev, subject: e.target.value }))}>
                    {contact.topics.map((topic) => <option key={topic} value={topic === contact.topics[0] ? '' : topic}>{topic}</option>)}
                  </select>
                </div>
                <div className="fgrp">
                  <label className="flbl">{contact.message}</label>
                  <textarea className="ftxt" rows={4} placeholder={contact.messagePlaceholder} value={contactForm.message} onChange={(e) => setContactForm((prev) => ({ ...prev, message: e.target.value }))} />
                </div>
                <button className="btn btn-g btn-full" onClick={submitContact} disabled={!contactForm.name || !contactForm.message}>📩 {contact.submit}</button>
                <div style={{ fontSize: 11, color: 'var(--tx4)', textAlign: 'center', marginTop: 8 }}>🔒 {contact.privacy}</div>
              </>
            )}
          </div>
        </div>
      </section>

      <section style={{ padding: '60px 28px', background: 'var(--g1)', textAlign: 'center' }}>
        <div style={{ maxWidth: 680, margin: '0 auto' }}>
          <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 34, fontWeight: 900, color: 'white', marginBottom: 10 }}>🌱 {cta.title}</div>
          <div style={{ fontSize: 15, color: 'rgba(255,255,255,.82)', marginBottom: 26, lineHeight: 1.75 }}>{cta.subtitle}</div>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
            <button className="btn btn-xl" style={{ padding: '13px 34px', background: 'var(--g5)', color: 'white', border: 'none', fontSize: 15, borderRadius: 12 }} onClick={() => nav('consultation')}>🔬 {cta.primary}</button>
            <button className="btn btn-xl" style={{ padding: '13px 34px', background: 'transparent', color: 'white', border: '1.5px solid rgba(255,255,255,.4)', fontSize: 15, borderRadius: 12 }} onClick={() => { setAuthMode('register'); setAuth(true); }}>📝 {cta.secondary}</button>
          </div>
          <div style={{ marginTop: 16, fontSize: 12, color: 'rgba(255,255,255,.5)' }}>{cta.note}</div>
        </div>
      </section>

      <footer className="footer">
        <div style={{ maxWidth: 1160, margin: '0 auto' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr', gap: 36, marginBottom: 36 }}>
            <div>
              <div style={{ fontFamily: "'Baloo 2',cursive", fontSize: 24, fontWeight: 900, marginBottom: 10 }}>🌱 BeejHealth</div>
              <div style={{ fontSize: 13.5, opacity: 0.72, lineHeight: 1.75, marginBottom: 18, maxWidth: 260 }}>{footer.description}</div>
              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 10, opacity: 0.45, textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: 7 }}>{footer.certifiedBy}</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {['ICAR', 'IARI', 'NABARD', 'Startup India'].map((item) => <span key={item} style={{ fontSize: 10, padding: '2px 9px', border: '1px solid rgba(255,255,255,.2)', borderRadius: 100, opacity: 0.65 }}>{item}</span>)}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 9 }}>
                {['📷', '🐦', '💼', '📘', '▶️'].map((icon) => <div key={icon} style={{ width: 34, height: 34, borderRadius: 8, background: 'rgba(255,255,255,.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: 14 }}>{icon}</div>)}
              </div>
            </div>
            {footer.columns.map((column) => (
              <div key={column.title}>
                <div style={{ fontSize: 10, fontWeight: 800, textTransform: 'uppercase', letterSpacing: 0.8, opacity: 0.45, marginBottom: 12 }}>{column.title}</div>
                {column.items.map((item) => <div key={item} style={{ fontSize: 13, opacity: 0.68, cursor: 'pointer', marginBottom: 8 }}>{item}</div>)}
              </div>
            ))}
          </div>
          <div style={{ borderTop: '1px solid rgba(255,255,255,.1)', paddingTop: 18, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
            <div style={{ fontSize: 12, opacity: 0.5 }}>{footer.copyright}</div>
            <div style={{ fontSize: 12, opacity: 0.5 }}>{footer.madeWith}</div>
          </div>
        </div>
      </footer>
    </>
  );
}
